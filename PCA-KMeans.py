import os
import numpy as np
from datetime import datetime
from information import dictionary

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, to_rgba

matplotlib.rcParams['image.cmap'] = 'winter'
matplotlib.rcParams['font.size'] = 22

mice = os.listdir('Data')
n_mice = len(mice)

timestamp_start = datetime.now()
for i_mouse, mouse in enumerate(mice):
    timestamp_mouse = datetime.now()
    print(f'\n{mouse} ({i_mouse+1} of {n_mice} mice)')

    # Create mouse folder
    try: os.mkdir(f'Figures_New/{mouse}')
    except: pass

# =============================================================================
#                     READING .RAW FILES
# =============================================================================

    files_directory = f'Data/{mouse}'

    listoffiles = os.listdir(files_directory)
    listoffiles = [file for file in listoffiles if '.raw' in file]

    listoffiles.sort(key=lambda x: round(float(x.split('_')[2].split('h')[0]) +
                     float(x.split('_')[2].split('h')[1].split('m')[0])/60 +
                     float(x.split('_')[2].split('h')[1].split('m')[1].split('s')[0])/3600, 5)*10e14 +
                     int(x.split('Frame')[1].split('-')[0])
                     )

    w = int(listoffiles[0].split('-_W')[1].split('_')[0])
    h = int(listoffiles[0].split(str(w) + '_H')[1].split('.raw')[0])

    frame_times = []
    frames = []

    for raw_file in listoffiles:
        with open(f'{files_directory}/{raw_file}', 'rb') as read_file:
            softwareVersion = read_file.read(32)
            currentFileName = read_file.read(128)
            previousFileName = read_file.read(128)
            nextFileName = read_file.read(128)
            chipNo = read_file.read(4)
            clockMulti = read_file.read(4)
            settlingClocks = read_file.read(4)
            startFrameNo = read_file.read(4)
            frame_time = np.frombuffer(read_file.read(4), dtype='<f')[0]
            frame_times.append(frame_time)
            read_file.read(27*4)

            bg = np.frombuffer(read_file.read(w*h*2), dtype='<h')

            for j in range(1024):
                FrameNo = read_file.read(4)
                FrameNo = np.frombuffer(FrameNo, dtype='uint32')
                read_file.read(15*4)
                Triggers = read_file.read(1)
                read_file.read(8*4)

                try:
                    framedata = np.frombuffer(read_file.read(w*h*2),
                                              dtype='<h')
                except: break

                try: framedata = np.flip(bg-framedata).reshape(h, w)
                except: break

                frames.append(framedata)

    if 1/np.array(frame_times).mean() < 8:
        frame_times = np.array(frame_times)/2
    else:
        frame_times = np.array(frame_times)

    fps = 1/frame_times.mean()
    frames = np.array(frames)

    stimulation, [FrameNo_list, cut_index] = dictionary[mouse]

    FrameNo_ref = FrameNo_list[:2]  # Start and end frame nos for reference
    FrameNo_stim = FrameNo_list[2:]  # Start and end frame nos for stimulation

    if FrameNo_stim[1] - FrameNo_stim[0] < 0:
        print(f'{mouse}. Stim frame error')
        continue

    # Remove unresponsive pixels
    top, left, right, bottom = cut_index

    mask_tb = list(range(top)) + list(range(h-bottom, h))
    mask_lr = list(range(left)) + list(range(w-right, w))
    h, w = h-top-bottom, w-left-right

    frames = np.delete(frames, mask_tb, axis=1)
    frames = np.delete(frames, mask_lr, axis=2).astype(float)

    n_frames = len(frames)

# =============================================================================
#                              PCA
# =============================================================================

    try: os.mkdir(f'Figures_New/{mouse}/PCA')
    except: pass
    export_dir = f'Figures_New/{mouse}/PCA/{mouse}'

    frames_scaled = frames.reshape((n_frames, h*w))
    frames_scaled = StandardScaler().fit_transform(frames_scaled)

    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(frames_scaled.T)

    explained_var = pca.explained_variance_ratio_
    explained_var_sum = np.cumsum(pca.explained_variance_ratio_)

    PC_values = np.arange(pca.n_components_) + 1
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(PC_values, explained_var, 'o-', linewidth=2, color='blue')
    ax.set_xlabel('Principal Component', fontsize=20)
    ax.set_ylabel('Variance Explained', fontsize=20)
    ax.set_xticks(PC_values)
    ax.set_title('Scree Plot', fontsize=25)
    ax1 = ax.twinx()
    ax1.plot(PC_values, explained_var_sum, 'x-', linewidth=2, color='red')
    ax1.set_ylabel('Variance Explained Cummulative')

    fig.savefig(f'{export_dir}_Scree plot.png')
    plt.close('all')

    # colors = cm.Set1(np.linspace(0, 1, 9))
    # colors = [colors for _ in range(11)]
    # colors = np.concatenate(colors)
    colors = ['tab:purple', 'tab:orange', 'tab:blue',
              'tab:red', 'tab:gray', 'tab:brown', 'tab:pink',
              'tab:olive', 'tab:cyan', 'k']
    colors = [to_rgba(color) for color in colors]
    colors = np.array(colors)

    k_clusters = 10
    silhouette_avg = []
    distortions = []
    for k_cluster in range(2, k_clusters+1, 1):
        kmeanModel = KMeans(n_clusters=k_cluster, n_init=10)
        kmeanModel.fit(principalComponents)
        cluster_labels = kmeanModel.labels_
        labels, counts = np.unique(cluster_labels, return_counts=True)

        # Sorting cluster by number of pixels
        cluster_labels_new = np.zeros_like(cluster_labels)
        for new_label, label in enumerate(np.flip(labels[np.argsort(counts)])):
            cluster_labels_new[cluster_labels == label] = new_label

        labels = cluster_labels_new.reshape(h, w) + 1

        silhouette_avg.append(silhouette_score(principalComponents,
                                               cluster_labels))

        colors_kmeans = colors[:k_cluster]
        cmap = ListedColormap(colors_kmeans)

        fig, [ax1, clrbr1] = plt.subplots(ncols=2, figsize=(15, 15))
        ax1.imshow(labels, interpolation='none', cmap=cmap)
        for each_label in range(labels.flatten().shape[0]):
            ax1.text(int(each_label % w),
                     int(each_label/w),
                     labels.flatten()[each_label],
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=7,
                     color='k')
        matplotlib.colorbar.Colorbar(clrbr1, mappable=ax1.images[0])
        clrbr1.set_yticks(np.arange(k_cluster))
        ax1.set_position([.08, .1, .8, .8])
        clrbr1.set_position([.92, .1, .02, .8])
        fig.savefig(f'{export_dir}_K means clustering_{k_cluster} clusters.png')
        plt.close('all')

        distortions.append(kmeanModel.inertia_)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(np.arange(2, k_clusters+1), silhouette_avg,
            'o-', linewidth=2, color='blue')
    ax.set_xlabel('K Clusters', fontsize=20)
    ax.set_ylabel('Silhouette Score', fontsize=20)
    ax.set_title('Silhouette Analysis', fontsize=25)
    fig.savefig(f'{export_dir}_Silhouette Score.png')

    distort_diff = [(distortions[i+1] - distortions[i])/distortions[i]
                    for i in range(0, k_cluster-2)]
    distort_diff = np.array(distort_diff)
    distort_diff = np.abs(np.insert(distort_diff, 0, 0))

    ax.clear()
    ax.plot(np.arange(2, k_clusters+1), distortions,
            'o-', linewidth=2, color='blue')
    ax.set_xlabel('K Clusters', fontsize=20)
    ax.set_ylabel('Distortion', fontsize=20)
    ax.set_title('Elbow Plot', fontsize=25)
    fig.savefig(f'{export_dir}_Elbow Plot.png')

    ax.clear()
    ax.hist(principalComponents.flatten(), bins=100)
    ax.set_xlabel('PC Value', fontsize=20)
    ax.set_ylabel('Count', fontsize=20)
    ax.set_title('PC Value Histogram', fontsize=25)
    fig.savefig(f'{export_dir}_Histogram.png')

    plt.close('all')

    print(f'{mouse} took {datetime.now() - timestamp_mouse}')
print(f'Total time: {datetime.now() - timestamp_start}')
