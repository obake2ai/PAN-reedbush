import numpy as np
import scipy
import os
import torch

from torch.autograd import Variable
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def generate_interpolation_video(generator, opt, image_shrink=1, image_zoom=1, duration_sec=60.0, smoothing_sec=1.0, mp4=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8):
    network_model = os.path.join(opt.loadDir, "generator_model_%s") % str(opt.resume - opt.resume%opt.modelsave_interval).zfill(8)
    if mp4 is None:
        mp4 = network_model + '-lerp.mp4'
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    if cuda:
        generator.cuda()
        generator = torch.nn.DataParallel(generator)

    print('Loading network from "%s"...' % network_model)
    generator.load_state_dict(torch.load(os.path.join(opt.loadDir, "generator_model_%s") % str(opt.resume - opt.resume%opt.modelsave_interval).zfill(8)))

    print('Generating latent vectors...')
    shape = [num_frames, opt.latent_dim] # [frame, component]
    all_latents = Variable(Tensor(random_state.randn(*shape).astype(np.float32)))
    all_latents = scipy.ndimage.gaussian_filter(z, [smoothing_sec * mp4_fps] + [0], mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        images = generator(latents)
        if image_zoom > 1:
            images = scipy.ndimage.zoom(images, [image_zoom, image_zoom, 1], order=0)
        if images.shape[0] == 1:
            images = images.repeat(3, 2) # grayscale => RGB
        return images

    # Generate video.
    import moviepy.editor # pip install moviepy
    result_subdir = opt.loadDir
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4), fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()
