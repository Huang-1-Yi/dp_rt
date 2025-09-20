import torch
from acdp.sampler.solver_utils import *

#----------------------------------------------------------------------------
# 初始化钩子函数以获取U-Net瓶颈层输出

def init_hook(net, class_labels=None):
    unet_enc_out = []
    def hook_fn(module, input, output):
        unet_enc_out.append(output.detach())
    if hasattr(net, 'guidance_type'):                                       # LDM和Stable Diffusion模型
        hook = net.model.model.diffusion_model.middle_block.register_forward_hook(hook_fn)
    elif net.img_resolution == 256:                                         # 分辨率为256的CM和ADM模型
        hook = net.model.middle_block.register_forward_hook(hook_fn)
    else:                                                                   # EDM模型
        module_name = '8x8_block2' if class_labels is not None else '8x8_block3'
        hook = net.model.enc[module_name].register_forward_hook(hook_fn)
    return unet_enc_out, hook

#----------------------------------------------------------------------------

def get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size):
    """
    自适应时间步选择:通过预测网络决定中间时间步tmid
    缩放因子预测:预测scale dir和scale time来调整模型输出
    多阶集成:支持1-4阶的求解器
    瓶颈特征利用:使用U-Net的瓶颈层特征作为预测输入
    """
    if hasattr(net, 'guidance_type') and net.guidance_type == 'classifier-free':
        unet_enc = torch.mean(unet_enc_out[-1], dim=1) if not use_afs else torch.zeros((2*batch_size, 8, 8), device=t_cur.device)
        output = AMED_predictor(unet_enc[batch_size:], t_cur, t_next)
    else:
        unet_enc = torch.mean(unet_enc_out[-1], dim=1) if not use_afs else torch.zeros((batch_size, 8, 8), device=t_cur.device)
        output = AMED_predictor(unet_enc, t_cur, t_next)
    output_list = [*output]
    
    if len(output_list) == 2:
        try:
            use_scale_time = AMED_predictor.module.scale_time
        except:
            use_scale_time = AMED_predictor.scale_time
        if use_scale_time:
            r, scale_time = output_list
            r = r.reshape(-1, 1, 1, 1)
            scale_time = scale_time.reshape(-1, 1, 1, 1)
            scale_dir = torch.ones_like(scale_time)
        else:
            r, scale_dir = output_list
            r = r.reshape(-极, 1, 1, 1)
            scale_dir = scale_dir.reshape(-1, 1, 1, 1)
            scale_time = torch.ones_like(scale_dir)
    elif len(output_list) == 3:
        r, scale_dir, scale_time = output_list
        r = r.reshape(-1, 1, 1, 1)
        scale_dir = scale_dir.reshape(-1, 1, 1, 1)
        scale_time = scale_time.reshape(-1, 1极, 1, 1)
    else:
        r = output.reshape(-1, 1, 1, 1)
        scale_dir = torch.ones_like(r)
        scale_time = torch.ones_like(r)
    return r, scale_dir, scale_time

#----------------------------------------------------------------------------
# 从预训练扩散模型获取去噪输出

def get_denoised(net, x, t, class_labels=None, condition=None, unconditional_condition=None):
    if hasattr(net, 'guidance_type'):     # LDM和Stable Diffusion模型
        denoised = net(x, t, condition=condition, unconditional_condition=unconditional_condition)
    else:
        denoised = net(x, t, class_labels=class_labels)
    return denoised

#----------------------------------------------------------------------------

def amed_sampler(
    net, 
    latents, 
    class_labels=None,
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial',
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False,
    AMED_predictor=None, 
    step_idx=None, 
    train=False, 
    **kwargs
):
    """
    AMED-Solver (https://arxiv.org/abs/2312.00094).

    参数:
        net: 封装的扩散模型
        latents: PyTorch张量，在时间`sigma_max`的输入样本
        class_labels: PyTorch张量，条件采样的条件或引导采样
        condition: PyTorch张量，用于LDM和Stable Diffusion模型的调节条件
        unconditional_condition: PyTorch张量，用于LDM和Stable Diffusion模型的无条件调节
        num_steps: `int`，带`num_steps-1`间隔的总时间步数
        sigma_min: `float`，采样结束时的sigma值
        sigma_max: `float`，采样开始时的sigma值
        schedule_type: `str`，时间调度类型。支持三种类型:
            - 'polynomial': 多项式时间调度（EDM推荐）
            - 'logsnr': 均匀logSNR时间调度（小分辨率数据集DPM-Solver推荐）
            - 'time_uniform': 均匀时间调度（高分辨率数据集DPM-Solver推荐）
            - 'discrete': LDM使用的时间调度（使用LDM和Stable Diffusion代码库的预训练扩散模型时推荐）
        schedule_rho: `float`，时间步指数。当schedule_type为['polynomial', 'time_uniform']时需要指定
        afs: `bool`，是否在采样开始时使用解析第一步（AFS）
        denoise_to_zero: `bool`，是否在采样结束时从`sigma_min`去噪到`0`
        return_inters: `bool`，是否保存中间结果（整个采样轨迹）
        AMED_predictor: 预测器网络
        step_idx: `int`，指定训练中采样步骤的索引
        train: `bool`，是否在训练循环中？
    返回:
        PyTorch张量。如果return_inters=True则返回生成样本或采样轨迹的批次
    """
    assert AMED_predictor is not None

    # 时间步离散化
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    
    # 主采样循环
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        unet_enc_out, hook = init_hook(net, class_labels)
        
        # Euler步
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur

        hook.remove()
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)
        r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size=latents.shape[0])
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur

        # 应用二阶校正
        denoised = get_denoised(net, x_next, scale_time * t_m极, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_mid = (x_next - denoised) / t_mid
        x_next = x_cur + scale_dir * (t_next - t_cur) * d_mid
    
        if return_inters:
            inters.append(x_next.unsqueeze(0))
        
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, [], [], r, scale_dir, scale_time
    return x_next

#----------------------------------------------------------------------------

def euler_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial',
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    AMED_predictor=None, 
    step_idx=None, 
    train=False, 
    **kwargs
):  
    """
    Euler采样器的AMED插件

    参数:
        net: 封装的扩散模型
        latents: PyTorch张量，在时间`sigma_max`的输入样本
        class_labels: PyTorch张量，条件采样的条件或引导采样
        condition: PyTorch张量，用于LDM和Stable Diffusion模型的调节条件
        unconditional_condition: PyTorch张量，用于LDM和Stable Diffusion模型的无条件调节
        num_steps: `int`，带`num_steps-1`间隔的总时间步数
        sigma_min: `float`，采样结束时的sigma值
        sigma_max: `float`，采样开始时的sigma值
        schedule_type: `str`，时间调度类型。支持三种类型:
            - 'polynomial': 多项式时间调度（EDM推荐）
            - 'logsnr': 均匀logSNR时间调度（小分辨率数据集DPM-Solver推荐）
            - 'time_uniform': 均匀时间调度（高分辨率数据集DPM-Solver推荐）
            - 'discrete': LDM使用的时间调度（使用LDM和Stable Diffusion代码库的预训练扩散模型时推荐）
        schedule_rho: `float`，时间步指数。当schedule_type为['polynomial', 'time_uniform']时需要指定
        afs: `bool`，是否在采样开始时使用解析第一步（AFS）
        denoise_to_zero: `bool`，是否在采样结束时从`sigma_min`去噪到`0`
        return_inters: `bool`，是否保存中间结果（整个采样轨迹）
        AMED_predictor: 预测器网络
        step_idx: `int`，指定训练中采样步骤的索引
        train: `bool`，是否在训练循环中？
    返回:
        PyTorch张量。如果return_inters=True则返回生成样本或采样轨迹的批次
    """

    # 时间步离散化
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # 主采样循环
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        if AMED_predictor is not None:
            unet_enc_out, hook = init_hook(net, class_labels)

        # Euler步
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
            
        if AMED_predictor is not None:
            hook.remove()
            t_cur = t_cur.reshape(-1, 1, 1, 1)
            t_next = t_next.reshape(-1, 1, 1, 1)
            r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size=latents.shape[0])
            t_mid = (t_next**r) * (t_cur**(1-r))
            x_next = x_cur + (t_mid - t_cur) * d_cur
        else:
            x_next = x_cur + (t_next - t_cur) * d_cur
        
        # 为学生模型添加额外步骤
        if AMED_predictor is not None:
            denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_mid = (x_next - denoised) / t_mid
            x_next = x_next + scale_dir * (t_next - t_mid) * d_mid
            
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, [], [], r, scale_dir, scale_time
    return x_next


#----------------------------------------------------------------------------

def ipndm_sampler(
    net, 
    latents, 
    class_labels=None, 
极    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial',
    schedule_rho=7, 
    afs=False,
    denoise_to_zero=False, 
    return_inters=False, 
    AMED_predictor=None, 
    train=False, 
    max_order=4, 
    buffer_model=[], 
    **kwargs
):
    """
    改进的PNDM采样器的AMED插件

    参数:
        net: 封装的扩散模型
        latents: PyTorch张量，在时间`sigma_max`的输入样本
        class_labels: PyTorch张量，条件采样的条件或引导采样
        condition: PyTorch张量，用于LDM和Stable Diffusion模型的调节条件
        unconditional_condition: PyTorch张量，用于LDM和Stable Diffusion模型的无条件调节
        num_steps: `int`，带`num_steps-1`间隔的总时间步数
        sigma_min: `float`，采样结束时的sigma值
        sigma_max: `float`，采样开始时的sigma值
        schedule_type: `str`，时间调度类型。支持三种类型:
            - 'polynomial': 多项式时间调度（EDM推荐）
            - 'logsnr': 均匀logSNR时间调度（小分辨率数据集DPM-Solver推荐）
            - 'time_uniform': 均匀时间调度（高分辨率数据集DPM-Solver推荐）
            - 'discrete': LDM使用的时间调度（使用LDM和Stable Diffusion代码库的预训练扩散模型时推荐）
        schedule_rho: `float`，时间步指数。当schedule_type为['polynomial', 'time_uniform']时需要指定
        afs: `bool`，是否在采样开始时使用解析第一步（AFS）
        denoise_to_zero: `bool`，是否在采样结束时从`sigma_min`去噪到`0`
        return_inters: `bool`，是否保存中间结果（整个采样轨迹）
        AMED_predictor: 预测器网络
        step_idx: `int`，指定训练中采样步骤的索引
        train: `bool`，是否在训练循环中？
        max_order: `int`，求解器的最大阶数。1 <= max_order <= 4
        buffer_model: `list`，历史模型输出
    返回:
        PyTorch张量。如果return_inters=True则返回生成样本或采样轨迹的批次
    """

    assert max_order >= 1 and max_order <= 4
    # 时间步离散化
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # 主采样循环
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    buffer_model = buffer_model if train else []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        if AMED_predictor is not None:
            unet_enc_out, hook = init_hook(net, class_labels)
        
        use_afs = (afs and len(buffer_model) == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        
        order = min(max_order, len(buffer_model)+1)
        if AMED_predictor is not None:
            hook.remove()
            t_cur = t_cur.reshape(-1, 1, 1, 1)
            t_next = t_next.reshape(-1, 1, 1, 1)
            r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size=latents.shape[0])
            t_mid = (t_next**r) * (t_cur**(1-r))
            if order == 1:      # 第一步Euler
                x_next = x_cur + (t_mid - t_cur) * d_cur
            elif order == 2:    # 使用一个历史点
                x_next = x_cur + (t_mid - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
            elif order == 3:    # 使用两个历史点
                x_next = x_cur + (t_mid - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
            elif order == 4:    # 使用三个历史点
                x_next = x_cur + (t_mid - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        else:
            if order == 1:      # 第一步Euler
                x_next = x_cur + (t_next - t_cur) * d_cur
            elif order == 2:    # 使用一个历史点
                x_next = x_cur + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
            elif order == 3:    # 使用两个历史点
                x_next = x_cur + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
            elif order == 4:    # 使用三个历史点
                x_next = x_cur + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())
        
        if AMED_predictor is not None:
            order = min(max_order, len(buffer_model)+1)
            denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_next - denoised) / t_mid
            if order == 1:      # 第一步Euler
                x_next = x_next + scale_dir * (t_next - t_mid) * d_cur
            elif order == 2:    # 使用一个历史点
                x_next = x_next + scale_dir * (t_next - t_mid) * (3 * d_cur - buffer_model[-1]) / 2
            elif order == 3:    # 使用两个历史点
                x_next = x_next + scale_dir * (t_next - t_mid) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
            elif order == 4:    # 使用三个历史点
                x_next = x_next + scale_dir * (t_next - t_mid) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
            
            if len(buffer_model) == max_order - 1:
                for k in range(max_order - 2):
                    buffer_model[k] = buffer_model[k+1]
                buffer_model[-1] = d_cur.detach()
            else:
                buffer_model.append(d_cur.detach())
                
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    
    if denoise极_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, buffer_model, [], r, scale_dir, scale_time
    return x_next

#----------------------------------------------------------------------------

def dpm_2_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial',
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    AMED_predictor=None, 
    step_idx=None, 
    train=False, 
    r=0.5, 
    **kwargs
):
    """
    DPM-Solver-2的AMED插件

    参数:
        net: 封装的扩散模型
        latents: PyTorch张量，在时间`sigma_max`的输入样本
        class_labels: PyTorch张量，条件采样的条件或引导采样
        condition: PyTorch张量，用于LDM和Stable Diffusion模型的调节条件
        unconditional_condition: PyTorch张量，用于LDM和Stable Diffusion模型的无条件调节
        num_steps: `int`，带`num_steps-1`间隔的总时间步数
        sigma_min: `float`，采样结束时的sigma值
        sigma_max: `float`，采样开始时的sigma值
        schedule_type: `str`，时间调度类型。支持三种类型:
            - 'polynomial': 多项式时间调度（EDM推荐）
            - 'logsnr': 均匀logSNR时间调度（小分辨率数据集DPM-Solver推荐）
            - 'time_uniform': 均匀时间调度（高分辨率数据集DPM-Solver推荐）
            - 'discrete': LDM使用的时间调度（使用LDM和Stable Diffusion代码库的预训练扩散模型时推荐）
        schedule_rho: `float`，时间步指数。当schedule_type为['polynomial', 'time_uniform']时需要指定
        afs: `bool`，是否在采样开始时使用解析第一步（AFS）
        denoise_to_zero: `bool`，是否在采样结束时从`sigma_min`去噪到`0`
        return_inters: `bool`，是否保存中间结果（整个采样轨迹）
        AMED_predictor: 预测器网络
        step_idx: `int`，指定训练中采样步骤的索引
        train: `bool`，是否在训练循环中？
        r: `float`，控制中间时间步位置的超参数。r=0.5恢复原始DPM-Solver-2
    返回:
        PyTorch张量。如果return_inters=True则返回生成样本或采样轨迹的批次
    """

    # 时间步离散化
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    
    # 主采样循环
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        if AMED_predictor is not None:
            unet_enc_out, hook = init_hook(net, class_labels)
        
        # Euler步
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur

        scale_time, scale_dir = 1, 1
        if AMED_predictor is not None:
            hook.remove()
            t_cur = t_cur.reshape(-1, 1, 1, 1)
            t_next = t_next.reshape(-1, 1, 1, 1)
            r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size=latents.shape[0])
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur

        # 应用二阶校正
        denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_mid = (x_next - denoised) / t_mid
        x_next = x_cur + scale_dir * (t_next - t_cur) * ((1 / (2 * r)) * d_mid + (1 - 1 / (2 * r)) * d_cur)
    
        if return_inters:
            inters.append(x_next.unsqueeze(0))
        
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, [], [], r, scale_dir, scale_time
    return x_next

#----------------------------------------------------------------------------

def dpm_pp_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    AMED_predictor=None, 
    step_idx=None, 
    train=False, 
    buffer_model=[], 
    buffer_t=[], 
    max_order=3, 
    predict_x0=True, 
    lower_order_final=True,
    **kwargs
):
    """
    多步DPM-Solver++的AMED插件

    参数:
        net: 封装的扩散模型
        latents: PyTorch张量，在时间`sigma_max`的输入样本
        class_labels: PyTorch张量，条件采样的条件或引导采样
        condition: PyTorch张量，用于LDM和Stable Diffusion模型的调节条件
        unconditional_condition: PyTorch张量，用于LDM和Stable Diffusion模型的无条件调节
        num_steps: `int`，带`num_steps-1`间隔的总时间步数
        sigma_min: `float`，采样结束时的sigma值
        sigma_max: `float`，采样开始时的sigma值
        schedule_type: `str`，时间调度类型。支持三种类型:
            - 'polynomial': 多项式时间调度（EDM推荐）
            - 'logsnr': 均匀logSNR时间调度（小分辨率数据集DPM-Solver推荐）
            - 'time_uniform': 均匀时间调度（高分辨率数据集DPM-Solver推荐）
            - 'discrete': LDM使用的时间调度（使用LDM和Stable Diffusion代码库的预训练扩散模型时推荐）
        schedule_极rho: `float`，时间步指数。当schedule_type为['polynomial', 'time_uniform']时需要指定
        afs: `bool`，是否在采样开始时使用解析第一步（AFS）
        denoise_to_zero: `bool`，是否在采样结束时从`sigma_min`去噪到`0`
        return_inters: `bool`，是否保存中间结果（整个采样轨迹）
        AMED_predictor: 预测器网络
        step_idx: `int`，指定训练中采样步骤的索引
        train: `bool`，是否在训练循环中？
        buffer_model: `list`，历史模型输出
        buffer_t: `list`，历史时间步
        max_order: `int`，求解器的最大阶数。1 <= max_order <= 3
        predict_x0: `bool`，是否使用数据预测公式
        lower_order_final: `bool`，是否在采样最后阶段降低阶数
    返回:
        PyTorch张量。如果return_inters=True则返回在时间`sigma_min`的样本或整个采样轨迹
    """

    assert max_order >= 1 and max_order <= 3
    # 时间步离散化
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # 主采样循环
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    buffer_model = buffer_model if train else []
    buffer_t = buffer_t if train else []
    if AMED_predictor is not None:
        num_steps = 2 * AMED_predictor.module.num_steps - 1 if train else 2 * num_steps - 1
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        if AMED_predictor is not None:
            step_cur = (2 * step_idx + 1 if train else 2 * i + 1)
            unet_enc_out, hook = init_hook(net, class_labels)
        else:
            step_cur = i + 1
        
        use_afs = (afs and len(buffer_model) == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
            denoised = x_cur - t_cur * d_cur
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
            
        buffer_model.append(dynamic_thresholding_fn(denoised)) if predict_x0 else buffer_model.append(d_cur)
        if AMED_predictor is not None:
            hook.remove()
            t_cur = t_cur.reshape(-1, 1, 1, 1)
            t_next = t_next.reshape(-1, 1, 1, 1)
            r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size=latents.shape[0])
            t_mid = (t_next**r) * (t_cur**(1-r))
        buffer_t.append(t_cur)
        
        t_next_temp = t_mid if AMED_predictor is not None else t_next
        if lower_order_final:
            order = step_cur if step_cur < max_order else min(max_order, num_steps - step_cur)
        else:
            order = min(max_order, step_cur)
        x_next = dpm_pp_update(x_cur, buffer_model, buffer_t, t_next_temp, order, predict_x0=predict_x0)
            
        # 为步骤指导添加额外步骤
        if AMED_predictor is not None:
            step_cur = step_cur + 1
            denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            model_out = dynamic_thresholding_fn(denoised) if predict_x0 else ((x_next - denoised) / t_mid)
            buffer_model.append(model_out)
            buffer_t.append(t_mid)
            
            if lower_order_final:
                order = step_cur if step_cur < max_order else min(max_order, num_steps - step_cur)
            else:
                order = min(step_cur, max_order)
            x_next = dpm_pp_update(x_next, buffer_model, buffer_t, t_next, order, predict_x0=predict_x0, scale=scale_dir)
            
        if len(buffer_model) >= 3:
            buffer_model = [a.detach() for a in buffer_model[-3:]]
            buffer_t = [a.detach() for a in buffer_t[-3:]]
        else:
            buffer_model = [a.detach() for a in buffer_model]
            buffer_t = [a.detach() for a in buffer_t]
        
        if return_inters:
            inters.append(x_next.unsqueeze(0))
            
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
            
    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, buffer_model, buffer_t, r, scale_dir, scale_time
    return x_next

#----------------------------------------------------------------------------

def heun_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    **kwargs
):
    """
    Heun二阶采样器。来自EDM: https://arxiv.org/abs/2206.00364.

    参数:
        net: 封装的扩散模型
        latents: PyTorch张量，在时间`sigma_max`的输入样本
        class_labels: PyTorch张量，条件采样的条件或引导采样
        condition: PyTorch张量，用于LDM和Stable Diffusion模型的调节条件
        unconditional_condition: PyTorch张量，用于LDM和Stable Diffusion模型的无条件调节
        num_steps: `int`，带`num_steps-1`间隔的总时间步数
        sigma_min: `float`，采样结束时的sigma值
        sigma_max: `float`，采样开始时的sigma值
        schedule_type: `str`，时间调度类型。支持三种类型:
            - 'polynomial': 多项式时间调度（EDM推荐）
            - 'logsnr': 均匀logSNR时间调度（小分辨率数据集DPM-Solver推荐）
            - 'time_uniform': 均匀时间调度（高分辨率数据集DPM-Solver推荐）
            - 'discrete': LDM使用的时间调度（使用LDM和Stable Diffusion代码库的预训练扩散模型时推荐）
        schedule_rho: `float`，时间步指数。当schedule_type为['polynomial', 'time_uniform']时需要指定
        afs: `bool`，是否在采样开始时使用解析第一步（AFS）
        denoise_to_zero: `bool`，是否在采样结束时从`sigma_min`去噪到`0`
        return_inters: `bool`，是否保存中间结果（整个采样轨迹）
    返回:
        PyTorch张量。如果return_inters=True则返回生成样本或采样轨迹的批次
    """

    # 时间步离散化
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho)

    # 主采样循环
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next

        # Euler步
        use_afs = (afs and i == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        x_next = x极_cur + (t_next - t_cur) * d_cur

        # 应用二阶校正
        denoised = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_prime = (x_next - denoised) / t_next
        x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

