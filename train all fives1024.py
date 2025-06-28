import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
from torch.nn import functional as F
from uutils import DiceLoss
from torch.nn.modules.loss import CrossEntropyLoss
from skimage import io
import torch.optim as optim
from medpy import metric
# set seeds
from scipy.ndimage.interpolation import zoom
torch.manual_seed(2023)
np.random.seed(2023)


def check(res1, gt):
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    res = res1
    gt = gt.squeeze().squeeze()
    # res = arrtools.extend1_before(res)
    # print(res.shape, gt.shape)
    # res = torch.from_numpy(res).float().to(device)
    # res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    # res = res.data.squeeze()
    # res = (res - res.min()) / (res.max() - res.min() + 1e-8)

    input = res
    target = np.array(gt)
    N = gt.shape
    smooth = 1
    b=0.0
    # print(input.shape, target.shape)
    input_flat = np.reshape(input,(-1))
    target_flat = np.reshape(target,(-1))

    intersection = (input_flat*target_flat)

    loss =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

    a =  '{:.4f}'.format(loss)
    a = float(a)
    b = b + a
    # print(a)
    return b
def mean_vol_dsc(vol_batch, gt_batch):
    val_acc = 0.0
    batch_size = vol_batch.shape[0]
    for i in range(batch_size): 
        vol = vol_batch[i,:,:] 
        gt_vol = gt_batch[i,:,:] 
        acc = check(vol, gt_vol) 
        val_acc += acc 

    val_acc = val_acc /batch_size
    return val_acc
class NpzDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        # self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
        self.imgs = np.vstack([d['imgs'] for d in self.npz_data])
        print(f"{self.ori_gts.shape=}, {self.imgs.shape=}")
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        # img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        imgs = self.imgs[index]
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 3))
        x_max = min(W, x_max + np.random.randint(0, 3))
        y_min = max(0, y_min - np.random.randint(0, 3))
        y_max = min(H, y_max + np.random.randint(0, 3))
        # x_min = 0
        # x_max = W
        # y_min = 0
        # y_max = H
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float(),imgs

        # %% test dataset class and dataloader


npz_tr_path = r''
work_dir = './work_dir'
task_name = ''
# prepare SAM model
model_type = 'vit_b'
checkpoint = 'work_dir/SAM/sam_vit_b_01ec64.pth'

device = 'cuda:0'
model_save_path = join(work_dir, task_name) 
os.makedirs(model_save_path, exist_ok=True)
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
sam_model.train()
b_lr = 0.0002/250
for p in sam_model.image_encoder.parameters():
    # p.requires_grad=True
    p.requires_grad=False
for p in sam_model.prompt_encoder.parameters():
    # p.requires_grad=True
    p.requires_grad=True
for p in sam_model.mask_decoder.parameters():
    # p.requires_grad=True
    p.requires_grad=True
for p in sam_model.mask_decoder2.parameters():
    # p.requires_grad=True
    p.requires_grad=True


for param in sam_model.image_encoder.LAFFM_module1.parameters(): 
    param.requires_grad = True 
for param in sam_model.image_encoder.LAFFM_module2.parameters(): 
    param.requires_grad = True 
for name, p in sam_model.image_encoder.named_parameters():
    if 'norm' in name.lower():
        p.requires_grad = True
    if 'pos_embed' in name.lower():
        p.requires_grad = True
 
params = ( 
    list(sam_model.mask_decoder.parameters()) + 
    list(sam_model.mask_decoder2.parameters()) + 
    list(sam_model.prompt_encoder.parameters()) + 
    [p for p in sam_model.image_encoder.parameters() if p.requires_grad] )
optimizer = optim.AdamW(params, lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
seg2_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes  

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        

        if weight is None:
            weight = [1.0] * (self.n_classes - 1)


        loss = 0.0

        for i in range(1, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i-1] 

        return loss / (self.n_classes - 1)
dice_loss = DiceLoss(2)


num_epochs = 150
losses = []

best_loss = 0
train_dataset = NpzDataset(npz_tr_path)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

npz_te_path = r''
test_dataset = NpzDataset(npz_te_path)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
ce_loss = CrossEntropyLoss()


iter_num=0
max_iterations = 250 * len(train_dataloader)

for epoch in range(num_epochs):
    epoch_loss = 0
    val_loss = 0
    accum_counter = 0
    optimizer.zero_grad()
    # train
    sam_model.train()
    for step, (gt2D, boxes, imgs) in enumerate(tqdm(train_dataloader)):

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        img_embeddings_list = []
        for img in imgs:
            sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
            img = np.array(img)
            resize_img = sam_transform.apply_image(img)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
            assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), \
                "input image should be resized to 1024*1024"
            embeddings = sam_model.image_encoder(input_image)
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]
            img_embeddings_list.append(embeddings)
        image_embedding = torch.cat(img_embeddings_list, dim=0).float()  # (B, C, H, W)

        low_res_masks, iou_predictions, msk_feat, up_embed = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,   # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,       # (B, 256, 64, 64)
            multimask_output=False,
            gt=gt2D,
            mode='train'
        )
        ps_mask = F.interpolate(low_res_masks, size=(int(low_res_masks.shape[-2]/4),
                                                     int(low_res_masks.shape[-1]/4)),
                                mode='bilinear')
        img_noise_gaussian = torch.randn(image_embedding.size()).cuda() * 0.2 * (
                                image_embedding.max() - image_embedding.min())
        image_embedding = (image_embedding.cuda() + img_noise_gaussian.cuda())

        low_res_masks2, iou_predictions2, attn1 = sam_model.mask_decoder2(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,    # (B, 256, 64, 64)
            multimask_output=False,
            mask_feat=ps_mask, 
            gt=gt2D,
            mode='train',
            msk_feat=msk_feat,
            up_embed=up_embed
        )

        sgt2D = torch.squeeze(gt2D, 1).cuda()
        sgt2D2 = torch.squeeze(gt2D, 1).cpu()
        low_res_label = zoom(sgt2D2, (1, 0.25, 0.25), order=0)
        low_res_label = torch.tensor(low_res_label).float().cuda()


        loss_ce1 = ce_loss(low_res_masks, low_res_label.long())
        loss_ce2 = ce_loss(low_res_masks2, sgt2D.long())
        

        loss1 = loss_ce1
        loss2 = loss_ce2
        weight = 0.6 ** (0.990 ** epoch)
        loss = (1 - weight) * loss1 + weight * loss2


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if iter_num < 250:
            lr_ = 0.0002 * ((iter_num + 1) / 250)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
        else:
            shift_iter = iter_num - 250
            assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
            lr_ = 0.0002* (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

        iter_num = iter_num + 1
    epoch_loss = epoch_loss/(step+1)
    torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_latest.pth'))


