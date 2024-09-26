import torch

def train_sam(args, net: nn.Module, optimizer, train_loader,
              epoch, writer, schedulers=None, vis=50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        # lossfunc = criterion_G
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            # del pack['pt']
            name = pack['image_meta_dict']['filename_or_obj']

            if args.prompt_approach == 'box':
                boxes = pack['box']
                box_torch = torch.as_tensor(boxes, dtype=torch.float32, device=GPUdevice)
                boxes = box_torch[None, :]

            else:
                if 'pt' not in pack:
                    imgs, pt, masks = generate_click_prompt(imgs, masks)
                    # pt, point_labels = get_ptsimgs(imgs)
                else:
                    pt = pack['pt']
                    point_labels = pack['p_label']

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                    imgs = imgs.repeat(1, 3, 1, 1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

                showp = pt

                ind += 1
                b_size, c, w, h = imgs.size()
                longsize = w if w >= h else h

                if point_labels[0] != None:  # != -1
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if args.prompt_approach == 'random_click':
                        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    elif args.prompt_approach == 'points_grids':
                        pass
                    pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                # true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.to(dtype=torch.float32, device=GPUdevice)

            if args.fine_tuning_configuration: 
                sam_no_freeze_block = [f"blocks.{idx}." for idx,i in enumerate(args.fine_tuning_configuration) if i == 1 ]

            
            '''Train'''
            for n, value in net.named_parameters():
                if "Adapter" not in n:
                    if args.fine_tuning_configuration:
                        if 1 not in [1 for val in sam_no_freeze_block if val in n]: 
                            value.requires_grad = False
                    else:
                        value.requires_grad = False


            imge = net.image_encoder(imgs)  # image embeddings

            if args.prompt_approach == 'box':
                se, de = net.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,
                )
            else:
                se, de = net.prompt_encoder(
                    points=pt,
                    boxes=None,
                    masks=None,
                )

            pred, _ = net.mask_decoder(  # batched predicted masks
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(),
                # get_dense_pe() get positional encoding used to encode point prompts
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
            )

            loss = lossfunc(pred, masks)  # pred -> mask  masks -> label

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.update()

    return loss


def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0, 0, 0, 0), (0, 0, 0, 0)
    rater_res = [(0, 0, 0, 0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        # lossfunc = criterion_G
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['label'].to(dtype=torch.float32, device=GPUdevice)

            name = pack['image_meta_dict']['filename_or_obj']

            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            if 'pt' not in pack:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']

            while (buoy + evl_ch) <= imgsw.size(-1):
                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch
                ind += 1

                if args.prompt_approach == 'box':
                    boxes = pack['box']
                    box_torch = torch.as_tensor(boxes, dtype=torch.float32, device=GPUdevice)
                    boxes = box_torch[None, :]
                    pass

                else:
                    pt = ptw
                    showp = pt
                    if point_labels[0] != None:
                        # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                        point_coords = pt
                        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                        if args.prompt_approach == 'random_click':
                            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                        elif args.prompt_approach == 'points_grids':
                            pass
                        pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    # true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype=torch.float32, device=GPUdevice)

                '''test'''
                with torch.no_grad():
                    imge = net.image_encoder(imgs)

                    if args.prompt_approach == 'box':
                        se, de = net.prompt_encoder(
                            points=None,
                            boxes=boxes,
                            masks=None,
                        )
                    else:
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )

                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                    )

                    tot += lossfunc(pred, masks)

                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Valid'
                        for na in name:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'

                        if args.prompt_approach == 'random_click':
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['valid_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, points=showp)
                        elif args.prompt_approach == 'points_grids':
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['valid_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, points=None)
                        elif args.prompt_approach == 'box':
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['valid_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, box=pack['box'])

                    mask_old = masks

                    temp = eval_seg(pred, mask_old, threshold)
                    print(temp)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot / n_val, tuple([a / n_val for a in mix_res])