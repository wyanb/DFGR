import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#import constants as const

__all__ = ["DFCN"]


def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv

def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes

class DFCN(nn.Module):
    def __init__(self, inplanes, outplanes, instrides, outstrides):
        super(DFCN, self).__init__()

        assert isinstance(inplanes, list)
        assert isinstance(outplanes, list) and len(outplanes) == 1
        assert isinstance(outstrides, list) and len(outstrides) == 1
        assert outplanes[0] == sum(inplanes)  # concat
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.instrides = instrides
        self.outstrides = outstrides
        self.scale_factors = [
            in_stride / outstrides[0] for in_stride in instrides
        ]  # for resize
        self.upsample_list = [
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            for scale_factor in self.scale_factors
        ]
        self.nf_fast_flow = nf_fast_flow(
                #[in_channels, int(input_size / scale), int(input_size / scale)],
                [272, int(14), int(14)],
                conv3x3_only=False,
                hidden_ratio=1.0,
                flow_steps=4,
            )

    def forward(self, input):
        features = input["features"]
        assert len(self.inplanes) == len(features)

        feature_list = []
        # resize & concatenate
        for i in range(len(features)):
            upsample = self.upsample_list[i]
            feature_resize = upsample(features[i])
            feature_list.append(feature_resize)

        feature_resize_level_list = []
        feature_resize1_list = []
        
        text_descriptions = ["X guide ","X gc","X g'"]
        features_norm_images = []
        
        #feature_guide_prom = feature_list[0]
        #torch.save(feature_guide_prom, './prompt/toothbrush.pt')
        feature_list[0] = torch.load('./prompt/zipper.pt')
        
        feature_guide = feature_list[0]
        feature_guide_single = feature_guide[0] #one picture feature (C H W)
        feature_guide_single = feature_guide_single.cpu()
        feature_guide_reshape = feature_guide_single.view(24, -1).T.numpy() #(Channel)
        pca = PCA(n_components = 2)
        features_reduced = pca.fit_transform(feature_guide_reshape)
        features_norm = np.linalg.norm(features_reduced, axis=1)
        features_norm_image = features_norm.reshape(14, 14)
        features_norm_images.append(features_norm_image)
        #plt.figure(figsize=(8,8))
        #plt.imshow(features_norm_image, cmap='viridis')
        #plt.colorbar()
        #plt.savefig('./prompt/toothbrush_prompt.png', bbox_inches='tight',pad_inches=0.0, dpi=300)
        ##plt.title("feature_guide PCA")
        #plt.show()
        
        for i1 in range(12):
            feature_resize1_list.append(feature_list[0])
        feature_resize_level1 = torch.cat(feature_resize1_list, dim=1)
        feature_resize_level1 = feature_resize_level1[:, 0:272, :, :]
        
        feature_g_single = feature_resize_level1[0] #one picture feature (C H W)
        feature_g_single = feature_g_single.cpu()
        feature_g_reshape = feature_g_single.view(272, -1).T.numpy() #(Channel)
        #pca = PCA(n_components = 2)
        features_g_reduced = pca.fit_transform(feature_g_reshape)
        features_g_norm = np.linalg.norm(features_g_reduced, axis=1)
        features_g_norm_image = features_g_norm.reshape(14, 14)
        features_norm_images.append(features_g_norm_image)
        #plt.figure(figsize=(8,8))
        #plt.imshow(features_g_norm_image, cmap='viridis')
        #plt.colorbar()
        #plt.title("feature_gc PCA")
        #plt.show()
        
        feature_resize_level1, log_jac_dets = self.nf_fast_flow(feature_resize_level1)
        
        feature_ff_single = feature_resize_level1[0] #one picture feature (C H W)
        feature_ff_single = feature_ff_single.cpu()
        feature_ff_reshape = feature_ff_single.view(272, -1).T.numpy() #(Channel)
        #pca = PCA(n_components = 2)
        features_ff_reduced = pca.fit_transform(feature_ff_reshape)
        features_ff_norm = np.linalg.norm(features_ff_reduced, axis=1)
        features_ff_norm_image = features_ff_norm.reshape(14, 14)
        features_norm_images.append(features_ff_norm_image)
        #plt.figure(figsize=(8,8))
        #plt.imshow(features_ff_norm_image, cmap='viridis')
        #plt.colorbar()
        #plt.title("feature_g' FF PCA")
        #plt.show()
        
        #plt.figure(figsize=(15,5))
        #for i in range(3):
        #    plt.subplot(1,3,i+1)
        #    plt.imshow(features_norm_images[i], cmap='viridis')
        #    plt.colorbar()
        #    plt.title("Feature Map {}".format(i+1))
        #    plt.text(0.5, -0.15, text_descriptions[i], ha='center', va='top', transform=plt.gca().transAxes, fontsize=12)
        #plt.subplots_adjust(left=0.03,right=0.985,top=0.98,bottom=0.1,wspace=0.05,hspace=0)
        #plt.show()
        
        
        feature_resize_level_list.append(feature_resize_level1)
        

        feature_resize2_list = []
        for i2 in range(9):
            feature_resize2_list.append(feature_list[1])
        feature_resize_level2 = torch.cat(feature_resize2_list, dim=1)
        feature_resize_level2 = feature_resize_level2[:, 0:272, :, :]
        feature_resize_level_list.append(feature_resize_level2)

        feature_resize3_list = []
        for i3 in range(5):
            feature_resize3_list.append(feature_list[2])
        feature_resize_level3 = torch.cat(feature_resize3_list, dim=1)
        feature_resize_level3 = feature_resize_level3[:, 0:272, :, :]
        feature_resize_level_list.append(feature_resize_level3)

        feature_align = torch.cat(feature_list, dim=1)

        #output, log_jac_dets = self.nf_fast_flow(features[0])

        return {"feature_align": feature_align, "outplane": self.get_outplanes(), "features": features, "feature_resize_level_list": feature_resize_level_list, "features_norm_images":features_norm_images}

    def get_outplanes(self):
        return self.outplanes

    def get_outstrides(self):
        return self.outstrides
