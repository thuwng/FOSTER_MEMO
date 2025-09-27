import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.foster import FOSTER, _KD_loss
from utils.inc_net import FOSTERNet
from utils.toolkit import count_parameters, tensor2numpy


class MEMO_FOSTER(FOSTER):
    """
    Kết hợp MEMO + FOSTER trong một learner duy nhất:
    - MEMO: mở rộng sâu theo nhánh (trên FOSTERNet) và đóng băng tầng nông sau base
      để giữ đặc trưng tổng quát; dùng exemplar rehearsal từ BaseLearner.
    - FOSTER: huấn luyện 2 pha cho mỗi task t>=1
        (1) Boosting: CE-bal(logits_total) + CE(fe_logits) + KD(oldfc→logits_total[:K_old])
        (2) Compression: chưng cất teacher→student bằng BKD (+ CE có trọng số),
            sau đó thay teacher bằng student để kiểm soát kích thước/ổn định.
    """
    def __init__(self, args):
        super().__init__(args)
        # MEMO+FOSTER: các tham số điều khiển
        # - memo_freeze_until: đóng băng tầng nông (MEMO-style) ở nhánh convnet mới
        # - kd_temperature, kd_alpha: nhiệt độ và trọng số KD cho giai đoạn nén (compression)
        # - memo_freeze: bật/tắt đóng băng kiểu MEMO; memo_bn_eval: BN eval ở phần đã freeze
        # - compression_ce_weight: trọng số CE trong compression; compression_lr: LR riêng cho compression
        self.memo_freeze_until = args.get('memo_freeze_until', 'layer2')
        self.kd_temperature = args.get('kd_temperature', args.get('T', 2))
        self.kd_alpha = args.get('kd_alpha', 1.0)
        self.memo_freeze = args.get('memo_freeze', True)
        self.memo_bn_eval = args.get('memo_bn_eval', True)
        self.compression_ce_weight = args.get('compression_ce_weight', 0.2)
        self.compression_lr = args.get('compression_lr', args.get('lr', 0.1))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self._cur_task > 1:
            self._network = self._snet
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        print('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # MEMO: Freeze shallow layers ở nhánh convnet mới (chỉ áp dụng cho incremental tasks)
        #  - CIFAR: 'stage_2' (conv_1_3x3, bn_1, stage_1, stage_2)
        #  - ImageNet: 'layer2' (conv1, bn1, layer1, layer2)
        if self._cur_task >= 1 and self.memo_freeze and self.args['init_cls'] >= 50:
            try:
                latest = self._network_module_ptr.convnets[-1]
                if hasattr(latest, 'freeze_until'):
                    latest.freeze_until(self.memo_freeze_until)
                if self.memo_bn_eval and hasattr(latest, 'set_bn_eval_until'):
                    try:
                        latest.set_bn_eval_until(self.memo_freeze_until)
                    except Exception as e2:
                        print(f"set_bn_eval_until skipped: {e2}")
            except Exception as e:
                print(f"freeze_until skipped: {e}")

        if self._cur_task > 0:
            # Khóa nhánh nền đầu tiên và oldfc (vai trò teacher trong FOSTER)
            for p in self._network.convnets[0].parameters():
                p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

        print('All params: {}'.format(count_parameters(self._network)))
        print('Trainable params: {}'.format(count_parameters(self._network, True)))

        # Trộn dữ liệu task hiện tại với exemplar (rehearsal) để giảm quên
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source='train',
            mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"],
                                       shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.args["batch_size"],
                                      shuffle=False, num_workers=self.args["num_workers"]) 

        device_ids = [d.index for d in self._multiple_gpus if isinstance(d, torch.device) and d.type == 'cuda' and d.index is not None and d.index < torch.cuda.device_count()]
        if len(device_ids) > 1:
            self._network = nn.DataParallel(self._network, device_ids=device_ids)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        # Unwrap only if actually wrapped
        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module

    def _feature_boosting(self, train_loader, test_loader, optimizer, scheduler):
        # FOSTER Boosting (áp dụng cho t>=1):
        #  - CE(logits_total/self.per_cls_weights, y): CE cân bằng theo tần suất lớp
        #  - CE(fe_logits, y): CE thường, buộc nhánh mới học đặc trưng lớp mới
        #  - KD(old_logits → logits_total[:K_old]; T=kd_temperature): giữ tri thức lớp cũ
        prog_bar = tqdm(range(self.args["boosting_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.
            losses_clf = 0.
            losses_fe = 0.
            losses_kd = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                outputs = self._network(inputs)
                logits, fe_logits, old_logits = outputs["logits"], outputs["fe_logits"], outputs["old_logits"].detach()
                loss_clf = F.cross_entropy(logits/self.per_cls_weights, targets)
                loss_fe = F.cross_entropy(fe_logits, targets)
                loss_kd = self.lambda_okd * _KD_loss(logits[:, :self._known_classes], old_logits, self.kd_temperature)
                loss = loss_clf+loss_fe+loss_kd
                optimizer.zero_grad()
                loss.backward()
                if self.oofc == "az":
                    for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                        if i == 0:
                            p.grad.data[self._known_classes:, :self._network_module_ptr.out_dim] = torch.tensor(0.0)
                elif self.oofc != "ft":
                    assert 0, "not implemented"
                optimizer.step()
                losses += loss.item()
                losses_fe += loss_fe.item()
                losses_clf += loss_clf.item()
                losses_kd += (self._known_classes / self._total_classes)*loss_kd.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch % 5 != 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["boosting_epochs"], losses/len(train_loader), losses_clf/len(train_loader), losses_fe/len(train_loader), losses_kd/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["boosting_epochs"], losses/len(train_loader), losses_clf/len(train_loader), losses_fe/len(train_loader), losses_kd/len(train_loader), train_acc)
            prog_bar.set_description(info)
            print(info)

    def _feature_compression(self, train_loader, test_loader):
        # FOSTER Compression (teacher -> student):
        #  - KD = BKD(stud_logits, teach_logits; T=kd_temperature) để cân bằng lớp theo FOSTER
        #  - + compression_ce_weight * CE(stud_logits, y) để neo student (tùy chọn, nhỏ)
        #  - Dùng compression_lr < lr boosting để nén ổn định; cuối pha, thay teacher bằng student
        self._snet = FOSTERNet(self.args['convnet_type'], False)
        self._snet.update_fc(self._total_classes)
        device_ids = [d.index for d in self._multiple_gpus if isinstance(d, torch.device) and d.type == 'cuda' and d.index is not None and d.index < torch.cuda.device_count()]
        if len(device_ids) > 1:
            self._snet = nn.DataParallel(self._snet, device_ids=device_ids)
        if hasattr(self._snet, "module"):
            self._snet_module_ptr = self._snet.module
        else:
            self._snet_module_ptr = self._snet
        self._snet.to(self._device)
        self._snet_module_ptr.convnets[0].load_state_dict(self._network_module_ptr.convnets[0].state_dict())
        self._snet_module_ptr.copy_fc(self._network_module_ptr.oldfc)

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._snet.parameters()),
                              lr=self.compression_lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["compression_epochs"]) 

        self._network.eval()
        prog_bar = tqdm(range(self.args["compression_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._snet.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                stud_logits = self._snet(inputs)["logits"]
                with torch.no_grad():
                    teach_out = self._network(inputs)
                    teach_logits = teach_out["logits"]
                # KD (Balanced KD) + optional CE có trọng số
                loss_kd = self.BKD(stud_logits, teach_logits, self.kd_temperature)
                loss_ce = F.cross_entropy(stud_logits, targets)
                loss = self.kd_alpha * loss_kd + self.compression_ce_weight * loss_ce
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(stud_logits[:targets.shape[0]], dim=1)
                correct += preds.eq(targets.expand_as(preds)).sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._snet, test_loader)
                info = 'SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["compression_epochs"], losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["compression_epochs"], losses/len(train_loader),  train_acc)
            prog_bar.set_description(info)
            print(info)

        if isinstance(self._snet, nn.DataParallel):
            self._snet = self._snet.module
        # Căn chỉnh trọng số student để giảm bias lớp cũ/mới
        if self.is_student_wa:
            self._snet.weight_align(self._known_classes, self._total_classes - self._known_classes, self.wa_value)
        else:
            print("do not weight align student!")
        self._snet.eval()
        # Swap to student
        self._snet_module_ptr = self._snet
        self._network = self._snet
        self._network_module_ptr = self._snet_module_ptr

