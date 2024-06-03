import torch



def gen_gaussian_hmap_op(coords, raw_size=(260,210), map_size=None, sigma=1, threshold=0, **kwargs):
    # openpose version
    # pose [T,18,3]; face [T,70,3]; hand_0(left) [T,21,3]; hand_1(right) [T,21,3]
    # gamma: hyper-param, control the width of gaussian, larger gamma, SMALLER gaussian
    # flags: use pose or face or hands or some of them

    #coords T, C, 3

    T, hmap_num = coords.shape[:2] 
    raw_h, raw_w = raw_size #260,210
    if map_size==None:
        map_h, map_w = raw_h, raw_w
        factor_h, factor_w = 1, 1
    else:
        map_h, map_w = map_size
        factor_h, factor_w = map_h/raw_h, map_w/raw_w
    # generate 2d coords
    # NOTE: openpose generate opencv-style coordinates!
    coords_y =  coords[..., 1]*factor_h
    coords_x = coords[..., 0]*factor_w
    confs = coords[..., 2] #T, C
    
    # if not limb:
    y, x = torch.meshgrid(torch.arange(map_h), torch.arange(map_w))
    coords = torch.stack([coords_y, coords_x], dim=0)  
    grid = torch.stack([y,x], dim=0).to(coords.device)  #[2,H,W]
    grid = grid.unsqueeze(0).unsqueeze(0).expand(hmap_num,T,-1,-1,-1)  #[C,T,2,H,W]
    coords = coords.unsqueeze(0).unsqueeze(0).expand(map_h, map_w,-1,-1,-1).permute(4,3,2,0,1)  #[C,T,2,H,W]
    hmap = torch.exp(-((grid-coords)**2).sum(dim=2) / (2*sigma**2))  #[C,T,H,W]
    hmap = hmap.permute(1,0,2,3)  #[T,C,H,W]
    if threshold > 0:
        confs = confs.unsqueeze(-1).unsqueeze(-1) #T,C,1,1
        confs = torch.where(confs>threshold, confs, torch.zeros_like(confs))
        hmap = hmap*confs
    
    center = kwargs.pop('center', None)
    if center is not None:
        # generate shifted heatmaps
        rela_hmap_lst = []
        for cen in center:
            if cen == 'nose':
                cen_y = coords_y[..., 52]
                cen_x = coords_x[..., 52]
            elif cen == 'shoulder_mid':
                right_y, right_x = coords_y[..., 57], coords_x[..., 57]
                left_y, left_x = coords_y[..., 58], coords_x[..., 58]
                cen_y, cen_x = (right_y+left_y)/2., (right_x+left_x)/2.
            c_y = (coords_y - cen_y.unsqueeze(1) + map_h) / 2.
            c_x = (coords_x - cen_x.unsqueeze(1) + map_w) / 2.
            coords = torch.stack([c_y, c_x], dim=0)
            coords = coords.unsqueeze(0).unsqueeze(0).expand(map_h, map_w,-1,-1,-1).permute(4,3,2,0,1)
            rela_hmap = torch.exp(-((grid-coords)**2).sum(dim=2) / (2*sigma**2))
            rela_hmap = rela_hmap.permute(1,0,2,3)  #[T,C,H,W]
            rela_hmap_lst.append(rela_hmap)
            # print(cen)
        
        if kwargs.pop('rela_comb', False):
            rela_hmap = torch.cat(rela_hmap_lst, dim=1)
            hmap = torch.cat([hmap, rela_hmap], dim=1)
            hmap = hmap.view(T, len(center)+1, hmap_num, map_h, map_w).permute(0,2,1,3,4).reshape(T, hmap_num*(len(center)+1), map_h, map_w)
            # print('rela_comb')
        else:
            hmap = rela_hmap_lst[0]

    temp_merge = kwargs.pop('temp_merge', False)
    if temp_merge: # and not limb:
        hmap = hmap.amax(dim=0)  #[C,H,W]

    return hmap