function KITTI_vis_test_seq
    use_gpu = 1;
   
    %%
    if exist('../+caffe', 'dir')
        addpath('..');
    else
        error('Please run this demo from caffe/matlab/demo');
    end
    
    %% Set caffe mode
    if exist('use_gpu', 'var') && use_gpu
        caffe.set_mode_gpu();
        gpu_id = 0;  % we will use the first gpu in this demo
        caffe.set_device(gpu_id);
    else
        caffe.set_mode_cpu();
    end
    
    
    net_model = 'scripts/deploy_resnet50by2_pool.prototxt';
    net_weights = 'model/train_iter_40000.caffemodel';

 
    phase = 'test';
    if ~exist(net_model, 'file')
        error('Please download CaffeNet from Model Zoo before you run this demo');
    end
    
    net = caffe.Net(net_model, net_weights,phase);
       
    prediction_disp = [];
    for i = 1:4
    im = imread(sprintf('images/%d.png',i));
    im = imresize(im,[160,608]);

    im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single    
    mean_data = repmat(reshape([104 117 123],1,1,3),size(im_data,1),size(im_data,2),1);
    im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)
        
    net.blobs('imL').reshape([size(im_data) 1]);
    
    tic;
    scores = net.forward({im_data});
    time_taken(i) = toc;
    
    subplot(2,1,1),imshow(im), title('Left image (Input)');
%     figure(1),subplot(2,1,4),imshow(imR),title('Right image');
%     figure(1),subplot(2,1,3),imshow(permute(scores{1}/255,[2,1,3])),title('back warp');
    A =  net.blobs('h_flow').get_data();
    figure(1),subplot(2,1,2), imagesc(permute(A,[2,1,3]));axis equal;axis off; axis tight;colormap gray,title('Depth');    
    pause(.5)
    
    prediction_disp(:,:,i) = permute(A,[2,1,3]);
    i
    end 
    caffe.reset_all();

    fprintf('Network running at %02g Hz',1/mean(time_taken(2:end)))
    
    %% processing  disparity to depth
    
    % scale the disparities to KITTI resolution
    scale = size(im,2)/size(prediction_disp,2);
    scale = permute(scale,[2 3 1]);
    flowHr =scale*imresize(prediction_disp,[370,1224]); 
    % estimate depth
    predicted_depths = 389.6304./flowHr;%predicted_depths = min(max(389.6304./flowHr,1),50);
    
    % load eigens mask and apply it for evaluation
    mask = logical(imread('images/mask_eigen.png'));
    
    for i = 1:size(predicted_depths,3)
        p = predicted_depths(:,:,i);
        p(~mask(:)) = nan;
        predicted_depths(:,:,i) = p; 
    end
    % evaluate disparity with your own evaluation function
    %A = evaluate_testset_kitti(predicted_depths);
     
1;