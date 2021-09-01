# 
# demo.py 
# 
import argparse
import os
import numpy as np

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image

try:
    import accimage
except ImportError:
    accimage = None


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def to_pil_image(pic):
    """Convert a tensor or an ndarray to PIL Image.

    See ``ToPIlImage`` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    mode = None
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))
    assert isinstance(npimg, np.ndarray)
    if npimg.shape[2] == 1:
        npimg = npimg[:, :, 0]

        if npimg.dtype == np.uint8:
            mode = 'L'
        if npimg.dtype == np.int16:
            mode = 'I;16'
        if npimg.dtype == np.int32:
            mode = 'I'
        elif npimg.dtype == np.float32:
            mode = 'F'
    elif npimg.shape[2] == 4:
            if npimg.dtype == np.uint8:
                mode = 'RGBA'
    else:
        if npimg.dtype == np.uint8:
            mode = 'RGB'
    assert mode is not None, '{} is not supported'.format(npimg.dtype)
    return Image.fromarray(npimg, mode=mode)

def evaluate_solder_spread(img_path):
    # find the predicted area of solder spreading (1-black pixels)
    count=0
    black=0
    im = Image.open(img_path)
    #im=transforms.ToPILImage(grid_image)
    #im=to_pil_image(grid_image)
    for pixel in im.getdata():
        count+=1
        if pixel == (0, 0, 0):
            black+=1

    solder_spread=count-black
    print("Total amount of Pixels {}".format(count))
    print("Background Pixels: {}".format(black))

    return solder_spread


def predict_image(input_image):
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='model_final.pth.tar',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=True, 
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    model = DeepLab(num_classes=3,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    image = Image.open(input_image).convert('RGB')
    target = Image.open(input_image).convert('L')
    sample = {'image': image, 'label': target}
    tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

    model.eval()
    if args.cuda:
        image = image.cuda()
    with torch.no_grad():
        output = model(tensor_in)

    grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 
                            3, normalize=False, range=(0, 255))
    print("type(grid) is: ", type(grid_image))
    print("grid_image.shape is: ", grid_image.shape)
    out_path="result_image.png"
    save_image(grid_image, out_path)

    solder_spread=evaluate_solder_spread(out_path)

    #os.remove(out_path)

    return out_path , solder_spread



   #for inference run: python demo.py --in-path your_file --out-path your_dst_file