import torch
import wandb
import os
from torchsummary import summary

from load_data import Load_data
from load_data import Get_datasets_info

from torchvision.models import resnet18

from net import Net


class Train:
    def __init__(self, start_epoch, epoch, lr, batch_size, num_workers, model_filename):
        super(Train, self).__init__()
        # 训练相关的初始化
        print('\033[34m训练准备......\033[0m')
        # 训练过程可视化——wandb初始化
        wandb.init(project="植物病害", entity="yujunyu")

        # gpu是否可用——并打印gpu信息
        self.CUDA = torch.cuda.is_available()
        if self.CUDA:  # True
            print('gpu_info:')
            gpu_info = (f'cuda.is_available:{self.CUDA}\n'
                        f'cuda.device_count:{torch.cuda.device_count()}\n'
                        f'cuda.device_name:{torch.cuda.get_device_name(0)}\n'
                        f'cuda.current_device:{torch.cuda.current_device()}')
            print(f"\033[32m{gpu_info}\033[0m")

        # batch_size和num_workers
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 加载数据集——并打印数据集信息
        self.train_loader, self.val_loader, class_to_idx = Load_data(trainDir='dataset/train', valDir='dataset/val',
                                                                     num_workers=self.num_workers, batch_size=self.batch_size)

        print('datasets_info:')
        Get_datasets_info(train_loader=self.train_loader, val_loader=self.val_loader, class_to_idx=class_to_idx)

        # 加载网络——并打印网络结构
        self.model_filename = model_filename
        if os.path.exists(self.model_filename):
            print("加载本地已有模型")
            self.net = resnet18(pretrained=False)
            fc_features = self.net.fc.in_features
            self.net.fc = torch.nn.Linear(in_features=fc_features, out_features=3)  # 3分类 修改输出,用in_features得到该层的输入，重写这一层
            if self.CUDA:
                self.net.cuda()
            state = torch.load(self.model_filename)
            self.net.load_state_dict(state)
        else:
            print("加载模型结构")
            # self.net = resnet18(pretrained=True)
            # fc_features = self.net.fc.in_features
            # self.net.fc = torch.nn.Linear(in_features=fc_features, out_features=3)
            self.net = Net()
            if self.CUDA:
                self.net.cuda()
            print('net_structure:')
            summary(model=self.net, input_size=(3, 224, 224))
            print(f"\033[32m{self.net}\033[0m")

        # 断点续训
        self.start_epoch = start_epoch

        # 迭代轮数epoch
        self.epoch = epoch

        # 学习率learning rate
        self.lr = lr

        # 优化器
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # 学习率策略
        # https://blog.csdn.net/qq_36722887/article/details/118612991
        # https://www.cnblogs.com/peachtea/p/13532209.html
        # 策略1：ReduceLROnPlateau()
        # self.scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5,
        #                                                        verbose=True, threshold=0.0001, threshold_mode='rel',
        #                                                        cooldown=0, min_lr=0, eps=1e-10)
        # 策略2：StepLR()
        # self.scheduler_lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)  # 每20epoch——>lr*0.1，
        # 策略3：CosineAnnealingLR()
        self.scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch, eta_min=0, last_epoch=-1)

        print("初始化的学习率：", self.optimizer.defaults['lr'])

        # 损失函数——交叉熵损失函数
        self.loss_function = torch.nn.CrossEntropyLoss()
        if self.CUDA:
            self.loss_function = self.loss_function.cuda()

    def training(self):
        print('\033[34m开始训练......\033[0m')
        best_acc = 0.0
        # 保存频率
        save_epoch = 5
        for e in range(self.start_epoch + 1, self.epoch + 1):
            self.net.train()  # 训练前加
            # 输入数量、预测正确数量
            inputs_num = 0.0
            correct_num = 0.0
            loss = 0.0
            for inputs, labels in self.train_loader:
                # 导数清零
                self.optimizer.zero_grad()
                if self.CUDA:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                # 计算输出
                outputs = self.net(inputs)
                # 计算损失——预测和实际的差值
                loss = self.loss_function(outputs, labels)
                # 反向传播
                loss.backward()
                # 更新梯度
                self.optimizer.step()

                # 计算精度
                # 转换为概率
                pre = torch.nn.functional.softmax(outputs, dim=1)
                # 输出预测类别
                pre = torch.argmax(pre, dim=1)
                # 累加预测正确的数量
                correct_num += (pre == labels).float().sum()
                # print(pre, labels, pre == labels)
                # 累加输入数量 50000
                inputs_num += len(inputs)

            # 更新学习率
            wandb.log({
                "Epoch": e,  # 加上epoch，可视化可以使step变epoch
                "Learning Rate": self.optimizer.param_groups[0]['lr']
            })
            print("第%d个epoch的学习率：%f" % (e, self.optimizer.param_groups[0]['lr']))
            self.scheduler_lr.step()

            train_acc = (correct_num / inputs_num) * 100.0
            # print(train_acc, loss)

            # 使用测试集验证
            val_acc, val_loss = self.validating()

            # 打印训练过程
            print(
                f"epoch:{e}/{epoch} \t train_acc:{train_acc} \t val_acc:{val_acc} \t train_loss:{loss} \t val_loss:{val_loss}")

            # wandb可视化
            wandb.log({
                "Train Accuracy": train_acc,
                "Val Accuracy": val_acc,
                # "Accuracy": {"Train:": train_accuracy, "Test": val_accuracy},
                "Train Loss": loss,
                "Val Loss": val_loss,
                "Epoch": e,  # 加上epoch，可视化可以使step变epoch
                # "Learning Rate": self.lr,
                # "Learning Rate": self.optimizer.param_groups[0]['lr']
            })

            # 保存准确率最高的模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.net.state_dict(), os.path.join(wandb.run.dir, self.model_filename))  # wandb云端

            # # 根据save_epoch保存模型
            # if e % save_epoch == 0:
            #     # torch.save(self.net.state_dict(), self.model_filename)    # 本地
            #     torch.save(self.net.state_dict(), os.path.join(wandb.run.dir, self.model_filename))  # wandb云端

        # 保存模型
        # torch.save(self.net.state_dict(), self.model_filename)    # 本地
        # torch.save(self.net.state_dict(), os.path.join(wandb.run.dir, self.model_filename))  # wandb云端
        # print(f'模型保存成功:{os.path.join(wandb.run.dir, self.model_filename)}')

    @torch.no_grad()
    def validating(self):
        self.net.eval()  # 测试前加
        inputs_num = 0.0
        correct_num = 0.0
        loss = 0.0
        for inputs, labels in self.val_loader:
            if self.CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)

            pre = torch.nn.functional.softmax(outputs, dim=1)
            pre = torch.argmax(pre, dim=1)
            correct_num += (pre == labels).float().sum()
            inputs_num += len(inputs)
        return (correct_num / inputs_num) * 100.0, loss


if __name__ == "__main__":
    # Hyper-parameter
    start_epoch = 0
    epoch = 20
    lr = 0.001
    batch_size = 128
    num_wokers = 0

    # Save weights file
    # 保存至本地
    # checkpoint_folder = 'checkpoints'
    # if not os.path.exists(checkpoint_folder):
    #     os.mkdir(checkpoint_folder)
    # model_filename = os.path.join(checkpoint_folder, 'test.pth')
    # 保存至wandb云端
    model_filename = 'model.pth'

    # Instantiate train
    trainer = Train(start_epoch, epoch, lr, batch_size, num_wokers, model_filename)

    # Execute trainingl
    trainer.training()
    print("训练完成")
