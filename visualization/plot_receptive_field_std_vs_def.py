import numpy as np
import matplotlib.pyplot as plt
import cv2

# Offsets data folder
offsets_prefix = 'c03-000c-01_offsets/c03-000c-01_offsets'

# Read text image
image = cv2.imread('c03-000c-01.png')
h, w, _ = image.shape
new_w = int(np.ceil(w*60/h))//2
image = cv2.resize(image, (new_w, 60))
layers_w = [new_w, new_w, new_w, new_w, new_w, new_w, new_w-1]
layers_h = [60, 60, 60, 60, 60, 60, 59]

plt.rcParams["figure.figsize"] = [int(np.ceil(w*2/h)), 2]

# COLLECT POINTS
last_depth = 6
counter = 0
for roi_idx in range(layers_w[last_depth]):
    for l_depth in reversed(range(last_depth+1)):
        current_layer = np.load(''.join([offsets_prefix, str(l_depth), '.npy'])).squeeze()
        print('Collecting from layer ', l_depth, 'of shape:', current_layer.shape)

        l_x = current_layer[1::2, :, :]
        l_y = current_layer[0::2, :, :]
        map_h = layers_h[l_depth]
        map_w = layers_w[l_depth]

        if l_depth == 6:
            filter_cells = [(0, 0), (0, 1),
                            (1, 0), (1, 1)]
        else:
            filter_cells = [(-1, -1), (-1, 0), (-1, 1),
                            (0,  -1), ( 0, 0), ( 0, 1),
                            (1,  -1), ( 1, 0), ( 1, 1)]

        if l_depth == last_depth:
            deeper_basepoints = []
            deeper_basepoints_standard = []
            for x in [roi_idx]:
                for y in range(map_h):
                    for c in range(current_layer.shape[0] // 2):
                        cell_y = filter_cells[c][0]
                        cell_x = filter_cells[c][1]
                        y_off = l_y[c, y, x]
                        x_off = l_x[c, y, x]
                        y_coord = np.round((y + cell_y + y_off) * layers_h[l_depth - 1] / map_h).astype('i')
                        x_coord = np.round((x + cell_x + x_off) * layers_w[l_depth - 1] / map_w).astype('i')
                        y_coord = min(y_coord, layers_h[l_depth - 1] - 1)
                        x_coord = min(x_coord, layers_w[l_depth - 1] - 1)
                        y_coord = max(y_coord, 0)
                        x_coord = max(x_coord, 0)
                        deeper_basepoints.append((y_coord, x_coord))
                        y_coord_standard = np.round((y + cell_y) * layers_h[l_depth - 1] / map_h).astype('i')
                        x_coord_standard = np.round((x + cell_x) * layers_w[l_depth - 1] / map_w).astype('i')
                        y_coord_standard = min(y_coord_standard, layers_h[l_depth - 1] - 1)
                        x_coord_standard = min(x_coord_standard, layers_w[l_depth - 1] - 1)
                        y_coord_standard = max(y_coord_standard, 0)
                        x_coord_standard = max(x_coord_standard, 0)
                        deeper_basepoints_standard.append((y_coord_standard, x_coord_standard))
        elif l_depth == 0:
            current_basepoints = set(deeper_basepoints)
            deeper_basepoints = []
            for _, current_basepoint in enumerate(current_basepoints):
                x = current_basepoint[1]
                y = current_basepoint[0]
                for c in range(current_layer.shape[0] // 2):
                    cell_y = filter_cells[c][0]
                    cell_x = filter_cells[c][1]
                    y_off = l_y[c, y, x]
                    x_off = l_x[c, y, x]
                    y_coord = np.round((y + cell_y + y_off)).astype('i')
                    x_coord = np.round((x + cell_x + x_off)).astype('i')
                    y_coord = min(y_coord, map_h - 1)
                    x_coord = min(x_coord, map_w - 1)
                    x_coord = max(x_coord, 0)
                    y_coord = max(y_coord, 0)
                    deeper_basepoints.append((y_coord, x_coord))
            current_basepoints_standard = set(deeper_basepoints_standard)
            deeper_basepoints_standard = []
            for _, current_basepoints_standard in enumerate(current_basepoints_standard):
                x = current_basepoints_standard[1]
                y = current_basepoints_standard[0]
                for c in range(current_layer.shape[0] // 2):
                    cell_y = filter_cells[c][0]
                    cell_x = filter_cells[c][1]
                    y_coord_standard = np.round((y + cell_y)).astype('i')
                    x_coord_standard = np.round((x + cell_x)).astype('i')
                    y_coord_standard = min(y_coord_standard, map_h - 1)
                    x_coord_standard = min(x_coord_standard, map_w - 1)
                    y_coord_standard = max(y_coord_standard, 0)
                    x_coord_standard = max(x_coord_standard, 0)
                    deeper_basepoints_standard.append((y_coord_standard, x_coord_standard))
        else:
            current_basepoints = set(deeper_basepoints)
            deeper_basepoints = []
            for _, current_basepoint in enumerate(current_basepoints):
                x = current_basepoint[1]
                y = current_basepoint[0]
                for c in range(current_layer.shape[0] // 2):
                    cell_y = filter_cells[c][0]
                    cell_x = filter_cells[c][1]
                    y_off = l_y[c, y, x]
                    x_off = l_x[c, y, x]
                    y_coord = np.round((y + cell_y + y_off) * layers_h[l_depth - 1] / map_h).astype('i')
                    x_coord = np.round((x + cell_x + x_off) * layers_w[l_depth - 1] / map_w).astype('i')
                    y_coord = min(y_coord, layers_h[l_depth - 1] - 1)
                    x_coord = min(x_coord, layers_w[l_depth - 1] - 1)
                    y_coord = max(y_coord, 0)
                    x_coord = max(x_coord, 0)
                    deeper_basepoints.append((y_coord, x_coord))
            current_basepoints_standard = set(deeper_basepoints_standard)
            deeper_basepoints_standard = []
            for _, current_basepoints_standard in enumerate(current_basepoints_standard):
                x = current_basepoints_standard[1]
                y = current_basepoints_standard[0]
                for c in range(current_layer.shape[0] // 2):
                    cell_y = filter_cells[c][0]
                    cell_x = filter_cells[c][1]
                    y_coord_standard = np.round((y + cell_y) * layers_h[l_depth - 1] / map_h).astype('i')
                    x_coord_standard = np.round((x + cell_x) * layers_w[l_depth - 1] / map_w).astype('i')
                    y_coord_standard = min(y_coord_standard, layers_h[l_depth - 1] - 1)
                    x_coord_standard = min(x_coord_standard, layers_w[l_depth - 1] - 1)
                    y_coord_standard = max(y_coord_standard, 0)
                    x_coord_standard = max(x_coord_standard, 0)
                    deeper_basepoints_standard.append((y_coord_standard, x_coord_standard))

    standard_x = []
    standard_y = []
    for tp in set(deeper_basepoints_standard):
        target_point = np.array(tp)
        standard_y.append(target_point[1])
        standard_x.append(target_point[0])
    deformed_x = []
    deformed_y = []
    for tp in set(deeper_basepoints):
        target_point = np.array(tp)
        deformed_y.append(target_point[1])
        deformed_x.append(target_point[0])
    counter += 1

    plt.imshow(image)
    plt.scatter(standard_y, standard_x, c='blue', alpha=0.3, marker='.', s=3)
    plt.scatter(deformed_y, deformed_x, c='red', alpha=0.3, marker='.', s=3)
    plt.tight_layout()
    plt.axis('off')
    plt.tight_layout(pad=0.0)
    plt.show(block=False)
    plt.pause(0.00001)
    plt.savefig('off_viz_results/receptive_field/img%04d.png' % counter)
    plt.clf()


