import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlockSM(nn.Module):
    """
    ResNet Block for ResNet18\ResNet34 models aka for smaller models

    """
    def __init__(self, dim_in, dim_out, expand_dims_without_stride=False, negative_slope=0.2):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.negative_slope = negative_slope
        if self.dim_in == self.dim_out:
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=1, bias=False)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
        else:
            stride = 2 if not expand_dims_without_stride else 1
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=stride, padding=1, bias=False)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
            self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=stride, padding=0, bias=False)
            nn.init.xavier_uniform_(self.conv_s.weight)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)

    def forward(self, input):
        output = self.conv_1(input)
        output = F.leaky_relu(output, negative_slope=self.negative_slope, inplace=True)
        output = self.conv_2(output)
        if self.dim_in == self.dim_out:
            output = output+input
        else:
            input_ = self.conv_s(input)
            output = output+input_
        output = F.leaky_relu(output, negative_slope=self.negative_slope, inplace=True)
        return output


class ResNetBlockSMBN(nn.Module):
    """
    ResNet Block for ResNet18\ResNet34 models aka for smaller models 
    with Batch Normalization layer

    """
    def __init__(self, dim_in, dim_out, expand_dims_without_stride=False, negative_slope=0.2):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.negative_slope = negative_slope
        if self.dim_in == self.dim_out:
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=1, bias=False)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
        else:
            stride = 2 if not expand_dims_without_stride else 1
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=stride, padding=1, bias=False)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
            self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=stride, padding=0, bias=False)
            nn.init.xavier_uniform_(self.conv_s.weight)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        self.bn1 = nn.BatchNorm2d(self.dim_out)
        self.bn2 = nn.BatchNorm2d(self.dim_out)

    def forward(self, input):
        output = self.conv_1(input)
        output = self.bn1(output)
        output = F.leaky_relu(output, negative_slope=self.negative_slope, inplace=True)
        output = self.conv_2(output)
        output = self.bn2(output)
        if self.dim_in == self.dim_out:
            output = output+input
        else:
            input_ = self.conv_s(input)
            output = output+input_
        output = F.leaky_relu(output, negative_slope=self.negative_slope, inplace=True)
        return output


class ResNetBlock(nn.Module):
    """
    ResNet Block for ResNet50\ResNet101\ResNet152 with full pre-activation mode
    From paper: Identity Mappings in Deep Residual Networks
                https://arxiv.org/pdf/1603.05027.pdf

    """
    def __init__(self, dim_in, dim_out, expand_dims_without_stride=False, negative_slope=0.2):
        super().__init__()
        self.dim_in = dim_in
        self.bottleneck_f = dim_out // 4
        self.dim_out = dim_out
        self.negative_slope = negative_slope
        if self.dim_in == self.dim_out:
            self.conv_1 = nn.Conv2d(self.dim_in, self.bottleneck_f, 1, stride=1, padding=0, bias=False)
            self.conv_2 = nn.Conv2d(self.bottleneck_f, self.bottleneck_f, 3, stride=1, padding=1, bias=False)
            self.conv_3 = nn.Conv2d(self.bottleneck_f, self.dim_out, 1, stride=1, padding=0, bias=False)
        else:
            stride = 2 if not expand_dims_without_stride else 1
            self.conv_1 = nn.Conv2d(self.dim_in, self.bottleneck_f, 1, stride=stride, padding=0, bias=False)
            self.conv_2 = nn.Conv2d(self.bottleneck_f, self.bottleneck_f, 3, stride=1, padding=1, bias=False)
            self.conv_3 = nn.Conv2d(self.bottleneck_f, self.dim_out, 1, stride=1, padding=0, bias=False)
            self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=stride, padding=0, bias=False)
            nn.init.xavier_uniform_(self.conv_s.weight)

        self.bn1 = nn.BatchNorm2d(self.dim_in)
        self.bn2 = nn.BatchNorm2d(self.bottleneck_f)
        self.bn3 = nn.BatchNorm2d(self.bottleneck_f)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)


    def forward(self, input):
        if self.dim_in == self.dim_out:
            output = self.bn1(input)
            output = F.leaky_relu(output, negative_slope=self.negative_slope, inplace=True)
            output = self.conv_1(output)
            output = self.bn2(output)
            output = F.leaky_relu(output, negative_slope=self.negative_slope, inplace=True)
            output = self.conv_2(output)
            output = self.bn3(output)
            output = F.leaky_relu(output, negative_slope=self.negative_slope, inplace=True)
            output = self.conv_3(output)
            output = output+input
        else:
            input = self.bn1(input)
            input = F.leaky_relu(input, negative_slope=self.negative_slope, inplace=True)
            output = self.conv_1(input)
            output = self.bn2(output)
            output = F.leaky_relu(output, negative_slope=self.negative_slope, inplace=True)
            output = self.conv_2(output)
            output = self.bn3(output)
            output = F.leaky_relu(output, negative_slope=self.negative_slope, inplace=True)
            output = self.conv_3(output)

            input_ = self.conv_s(input)
            output = output+input_
        return output


class ImageEncoderOriginal(nn.Module):
    """
    ImageEncoder used in paper Neural Template
    Keep it here for pre-trained original weights

    """
    def __init__(self, config):
        super(ImageEncoderOriginal, self).__init__()
        self.img_ef_dim = config.img_ef_dim
        # TODO: Remove or refactor below
        """
        self.z_dim = config.decoder_input_embbeding_size * 2 if config.decoder_type == 'Flow' and hasattr(config,
                                                                                                          'flow_use_split_dim') \
                                                                and config.flow_use_split_dim else config.decoder_input_embbeding_size
        """
        # Updated self.z_dim - just give output without any resctrickts
        self.z_dim = config.output_channels
        if hasattr(config, 'sample_class') and config.sample_class and self.z_dim / config.decoder_input_embbeding_size != 3:
            raise Exception(
                'Error! Sample class is expected but size of output encoder is not divided by 3 (number of modules). '
                f'z_dim={self.z_dim}, decoder_input_embbeding_size={config.decoder_input_embbeding_size}.'
            )
        elif (not hasattr(config, 'sample_class') or (hasattr(config, 'sample_class') and not config.sample_class)) \
                and self.z_dim / config.decoder_input_embbeding_size != 2:
            raise Exception(
                'Error! Output dim is not divided by 2 (number of modules). '
                f'z_dim={self.z_dim}, decoder_input_embbeding_size={config.decoder_input_embbeding_size}.'
            )
        if hasattr(config, 'image_preferred_color_space'):
            in_f = config.image_preferred_color_space
        else:
            in_f = 1
        
        if hasattr(config, 'use_depth') and config.use_depth:
            in_f += 1
        self.invert_input = config.image_invert_input if hasattr(config, 'image_invert_input') else True 
        self.in_f = in_f

        self.conv_0 = nn.Conv2d(self.in_f, self.img_ef_dim, 7, stride=2, padding=3, bias=False)
        self.res_1 = ResNetBlockSM(self.img_ef_dim, self.img_ef_dim, negative_slope=0.01)
        self.res_2 = ResNetBlockSM(self.img_ef_dim, self.img_ef_dim, negative_slope=0.01)
        self.res_3 = ResNetBlockSM(self.img_ef_dim, self.img_ef_dim * 2, negative_slope=0.01)
        self.res_4 = ResNetBlockSM(self.img_ef_dim * 2, self.img_ef_dim * 2, negative_slope=0.01)
        self.res_5 = ResNetBlockSM(self.img_ef_dim * 2, self.img_ef_dim * 4, negative_slope=0.01)
        self.res_6 = ResNetBlockSM(self.img_ef_dim * 4, self.img_ef_dim * 4, negative_slope=0.01)
        self.res_7 = ResNetBlockSM(self.img_ef_dim * 4, self.img_ef_dim * 8, negative_slope=0.01)
        self.res_8 = ResNetBlockSM(self.img_ef_dim * 8, self.img_ef_dim * 8, negative_slope=0.01)
        self.conv_9 = nn.Conv2d(self.img_ef_dim * 8, self.img_ef_dim * 16, 4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv2d(self.img_ef_dim * 16, self.img_ef_dim * 16, 4, stride=1, padding=0, bias=True)
        self.linear_1 = nn.Linear(self.img_ef_dim * 16, self.img_ef_dim * 16, bias=True)
        self.linear_2 = nn.Linear(self.img_ef_dim * 16, self.img_ef_dim * 16, bias=True)
        self.linear_3 = nn.Linear(self.img_ef_dim * 16, self.img_ef_dim * 16, bias=True)
        self.linear_4 = nn.Linear(self.img_ef_dim * 16, self.z_dim, bias=True)
        nn.init.xavier_uniform_(self.conv_0.weight)
        nn.init.xavier_uniform_(self.conv_9.weight)
        nn.init.constant_(self.conv_9.bias, 0)
        nn.init.xavier_uniform_(self.conv_10.weight)
        nn.init.constant_(self.conv_10.bias, 0)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias, 0)

    def forward(self, view):
        layer_0 = self.conv_0(1 - view if self.invert_input else view)
        layer_0 = F.leaky_relu(layer_0, negative_slope=0.01, inplace=True)

        layer_1 = self.res_1(layer_0)
        layer_2 = self.res_2(layer_1)

        layer_3 = self.res_3(layer_2)
        layer_4 = self.res_4(layer_3)

        layer_5 = self.res_5(layer_4)
        layer_6 = self.res_6(layer_5)

        layer_7 = self.res_7(layer_6)
        layer_8 = self.res_8(layer_7)

        layer_9 = self.conv_9(layer_8)
        layer_9 = F.leaky_relu(layer_9, negative_slope=0.01, inplace=True)

        layer_10 = self.conv_10(layer_9)
        layer_10 = layer_10.view(-1, self.img_ef_dim * 16)
        layer_10 = F.leaky_relu(layer_10, negative_slope=0.01, inplace=True)

        l1 = self.linear_1(layer_10)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

        l4 = self.linear_4(l3)
        l4 = torch.sigmoid(l4)

        return l4


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.img_ef_dim = config.img_ef_dim
        # Updated self.z_dim - just give output without any resctrickts
        self.z_dim = config.output_channels
        if hasattr(config, 'sample_class') and config.sample_class and self.z_dim / config.decoder_input_embbeding_size != 3:
            raise Exception(
                'Error! Sample class is expected but size of output encoder is not divided by 3 (number of modules). '
                f'z_dim={self.z_dim}, decoder_input_embbeding_size={config.decoder_input_embbeding_size}.'
            )
        elif (not hasattr(config, 'sample_class') or (hasattr(config, 'sample_class') and not config.sample_class)) \
                and self.z_dim / config.decoder_input_embbeding_size != 2:
            raise Exception(
                'Error! Output dim is not divided by 2 (number of modules). '
                f'z_dim={self.z_dim}, decoder_input_embbeding_size={config.decoder_input_embbeding_size}.'
            )
        if hasattr(config, 'image_preferred_color_space'):
            in_f = config.image_preferred_color_space
        else:
            in_f = 1
        
        if hasattr(config, 'use_depth') and config.use_depth:
            in_f += 1
        self.in_f = in_f
        self.invert_input = config.image_invert_input if hasattr(config, 'image_invert_input') else True 

        if hasattr(config, 'img_arch_type'):
            if config.img_arch_type.lower() == 'resnet18':
                self.model = ResNet18(
                    in_f, self.img_ef_dim, self.z_dim, 
                    config.type_block if hasattr(config, 'type_block') else 'ResNetBlockSMBN'
                )
            elif config.img_arch_type.lower() == 'resnet34':
                self.model = ResNet34(
                    in_f, self.img_ef_dim, self.z_dim, 
                    config.type_block if hasattr(config, 'type_block') else 'ResNetBlockSMBN'
                )
            elif config.img_arch_type.lower() == 'resnet50':
                self.model = ResNet50(
                    in_f, self.img_ef_dim, self.z_dim, 
                )
            else:
                raise Exception(f'Unknown architecture {config.img_arch_type}')
        else:
            self.model = ResNet18(
                in_f, self.img_ef_dim, self.z_dim, 
                config.type_block if hasattr(config, 'type_block') else 'ResNetBlockSMBN'
            )
        self.linear_1 = nn.Linear(self.model.final_out_f, self.model.final_out_f, bias=True)
        self.linear_2 = nn.Linear(self.model.final_out_f, self.model.final_out_f, bias=True)
        self.linear_3 = nn.Linear(self.model.final_out_f, self.model.final_out_f, bias=True)
        self.linear_4 = nn.Linear(self.model.final_out_f, self.z_dim, bias=True)
        self.final_act_func = config.img_final_act_func if hasattr(config, 'img_final_act_func') else torch.sigmoid
        if hasattr(config, 'img_linear_use_bn') and config.img_linear_use_bn:
            self.linear_use_bn = True
            self.bn1 = nn.BatchNorm1d(self.model.final_out_f)
            self.bn2 = nn.BatchNorm1d(self.model.final_out_f)
            self.bn3 = nn.BatchNorm1d(self.model.final_out_f)
        else:
            self.linear_use_bn = False
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias, 0)

    def forward(self, view):
        output = self.model(1 - view if self.invert_input else view)
        b, _, _, _ = output.shape
        output = output.view(b, self.model.final_out_f)
        output = F.leaky_relu(output, negative_slope=0.2, inplace=True)
        l1 = self.linear_1(output)
        if self.linear_use_bn:
            l1 = self.bn1(l1)
        l1 = F.leaky_relu(l1, negative_slope=0.2, inplace=True)

        l2 = self.linear_2(l1)
        if self.linear_use_bn:
            l2 = self.bn2(l2)
        l2 = F.leaky_relu(l2, negative_slope=0.2, inplace=True)

        l3 = self.linear_3(l2)
        if self.linear_use_bn:
            l3 = self.bn3(l3)
        l3 = F.leaky_relu(l3, negative_slope=0.2, inplace=True)

        l4 = self.linear_4(l3)
        if self.final_act_func is not None:
            l4 = self.final_act_func(l4)

        return l4
    

class ResNet18(nn.Module):
    def __init__(self, in_f, base_f, out_f, type_block='ResNetBlockSM'):
        super().__init__()  
        self.in_f = in_f      
        self.base_f = base_f
        self.out_f = out_f
        self.final_out_f = base_f * 8

        if type_block == ResNetBlockSM.__name__:
            resnet_block = ResNetBlockSM
            self.use_bn = False
        elif type_block == ResNetBlockSMBN.__name__:
            resnet_block = ResNetBlockSMBN
            self.use_bn = True
        else:
            raise Exception(f'Unknown input type block {type_block}.')

        if self.use_bn:
            self.bn_initial = nn.BatchNorm2d(in_f)
            # Scale (weight) should be turned off. Only `bias` is needed at the start
            # Notice that `weight` - is a scale, there is also `bias` which will be trained
            self.bn_initial.weight.requires_grad = False
            self.bn0 = nn.BatchNorm2d(base_f // 2)

        self.conv_0 = nn.Conv2d(in_f, base_f // 2, 7, stride=2, padding=3, bias=False)

        self.res_1 = resnet_block(base_f // 2, base_f, expand_dims_without_stride=True)
        self.res_2 = resnet_block(base_f, base_f)

        self.res_3 = resnet_block(base_f, base_f * 2)
        self.res_4 = resnet_block(base_f * 2, base_f * 2)

        self.res_5 = resnet_block(base_f * 2, base_f * 4)
        self.res_6 = resnet_block(base_f * 4, base_f * 4)

        self.res_7 = resnet_block(base_f * 4, self.final_out_f)
        self.res_8 = resnet_block(self.final_out_f, self.final_out_f)

    def forward(self, x):
        if self.use_bn:
            x = self.bn_initial(x)
        x = self.conv_0(x)
        if self.use_bn:
            x = self.bn0(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

        x = F.max_pool2d(x, (3,3), stride=2, padding=1)

        x = self.res_1(x)
        x = self.res_2(x)

        x = self.res_3(x)
        x = self.res_4(x)

        x = self.res_5(x)
        x = self.res_6(x)

        x = self.res_7(x)
        x = self.res_8(x)

        x = F.adaptive_avg_pool2d(x, (1,1))

        return x


class ResNet34(nn.Module):
    def __init__(self, in_f, base_f, out_f, type_block='ResNetBlockSM'):
        super().__init__()  
        self.in_f = in_f      
        self.base_f = base_f
        self.out_f = out_f
        self.final_out_f = base_f * 8

        if type_block == ResNetBlockSM.__name__:
            resnet_block = ResNetBlockSM
            self.use_bn = False
        elif type_block == ResNetBlockSMBN.__name__:
            resnet_block = ResNetBlockSMBN
            self.use_bn = True
        else:
            raise Exception(f'Unknown input type block {type_block}.')

        if self.use_bn:
            self.bn_initial = nn.BatchNorm2d(in_f)
            # Scale (weight) should be turned off. Only `bias` is needed at the start
            # Notice that `weight` - is a scale, there is also `bias` which will be trained
            self.bn_initial.weight.requires_grad = False
            self.bn0 = nn.BatchNorm2d(base_f // 2)
        
        self.conv_0 = nn.Conv2d(in_f, base_f // 2, 7, stride=2, padding=3, bias=False)
        
        self.resnet_block_1 = nn.Sequential(
            resnet_block(base_f // 2, base_f, expand_dims_without_stride=True), 
            resnet_block(base_f, base_f), 
            resnet_block(base_f, base_f),
        )
        self.resnet_block_2 = nn.Sequential(
            resnet_block(base_f, base_f * 2),
            resnet_block(base_f * 2, base_f * 2),
            resnet_block(base_f * 2, base_f * 2),
        )
        self.resnet_block_3 = nn.Sequential(
            resnet_block(base_f * 2, base_f * 4),
            resnet_block(base_f * 4, base_f * 4),
            resnet_block(base_f * 4, base_f * 4),

            resnet_block(base_f * 4, base_f * 4),
            resnet_block(base_f * 4, base_f * 4),
            resnet_block(base_f * 4, base_f * 4),
        )
        self.resnet_block_4 = nn.Sequential(
            resnet_block(base_f * 4, self.final_out_f),
            resnet_block(self.final_out_f, self.final_out_f),
            resnet_block(self.final_out_f, self.final_out_f),
        )

    def forward(self, x):
        if self.use_bn:
            x = self.bn_initial(x)
        x = self.conv_0(x)
        if self.use_bn:
            x = self.bn0(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

        x = F.max_pool2d(x, (3,3), stride=2, padding=1)

        x = self.resnet_block_1(x)
        x = self.resnet_block_2(x)
        x = self.resnet_block_3(x)
        x = self.resnet_block_4(x)

        x = F.adaptive_avg_pool2d(x, (1,1))

        return x


class ResNet50(nn.Module):
    def __init__(self, in_f, base_f, out_f):
        super().__init__()  
        self.in_f = in_f      
        self.base_f = base_f
        self.out_f = out_f
        self.final_out_f = base_f * 32

        self.bn_initial = nn.BatchNorm2d(in_f)
        # Scale (weight) should be turned off. Only `bias` is needed at the start
        # Notice that `weight` - is a scale, there is also `bias` which will be trained
        self.bn_initial.weight.requires_grad = False
        self.conv_0 = nn.Conv2d(in_f, base_f, 3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(base_f)
        self.conv_1 = nn.Conv2d(base_f, base_f, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_f)
        self.conv_2 = nn.Conv2d(base_f, base_f, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_f)

        self.resnet_block_1 = nn.Sequential(
            ResNetBlock(base_f, base_f * 4, expand_dims_without_stride=True), 
            ResNetBlock(base_f * 4, base_f * 4), 
            ResNetBlock(base_f * 4, base_f * 4),
        )
        self.resnet_block_2 = nn.Sequential(
            ResNetBlock(base_f * 4, base_f * 8),
            ResNetBlock(base_f * 8, base_f * 8),
            ResNetBlock(base_f * 8, base_f * 8),
        )
        self.resnet_block_3 = nn.Sequential(
            ResNetBlock(base_f * 8, base_f * 16),
            ResNetBlock(base_f * 16, base_f * 16),
            ResNetBlock(base_f * 16, base_f * 16),

            ResNetBlock(base_f * 16, base_f * 16),
            ResNetBlock(base_f * 16, base_f * 16),
            ResNetBlock(base_f * 16, base_f * 16),
        )
        self.resnet_block_4 = nn.Sequential(
            ResNetBlock(base_f * 16, self.final_out_f),
            ResNetBlock(self.final_out_f, self.final_out_f),
            ResNetBlock(self.final_out_f, self.final_out_f),
        )

    def forward(self, x):
        x = self.bn_initial(x)

        x = self.conv_0(x)
        x = self.bn0(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

        x = self.conv_1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

        x = self.conv_2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

        x = F.max_pool2d(x, (3,3), stride=2, padding=1)

        x = self.resnet_block_1(x)
        x = self.resnet_block_2(x)
        x = self.resnet_block_3(x)
        x = self.resnet_block_4(x)

        x = F.adaptive_avg_pool2d(x, (1,1))

        return x

