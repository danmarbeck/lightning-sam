import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import os
import pathlib
import pickle
import statistics
from pathlib import Path
from PIL import Image, ImageDraw
from helpers import *

def load_json_file(path_to_json_file):
    if os.path.isfile(path_to_json_file):
        with open(path_to_json_file) as file:
            json_objects = json.load(file)
            file.close()

        return json_objects

path_to_experiments = '/media/jacqueline/IcyBoxData/Paper/Gaze_Meets_ML_NIPS_2023/datasets/Cellpose_500/Cellpose_500_raw_data/Cellpose_500_zoom_no_timelimit/'
path_to_save_trainings_dataset = '/media/jacqueline/IcyBoxData/Paper/Gaze_Meets_ML_NIPS_2023/figures/dataset_figure/images/'
path_to_images = '/media/jacqueline/IcyBoxData/Paper/Gaze_Meets_ML_NIPS_2023/datasets/Cellpose_500/Cellpose_500_groundtruth_data/'
path_to_ready_dataset_heatmaps = '/media/jacqueline/IcyBoxData/Paper/Gaze_Meets_ML_NIPS_2023/datasets/Cellpose_500/Cellpose_500_ready_dataset/'


#color_scheme = {'background' : 0, 'dog' : 1, 'cat' : 2, 'bird' : 3, 'horse' : 4}
color_scheme = {'background' : 0,
              'aeroplane' : 1,
              'bicycle' : 2,
              'bird' : 3,
              'boat' : 4,
              'bottle' : 5,
              'bus' : 6,
              'car' : 7,
              'cat' : 8,
              'chair' : 9,
              'cow' : 10,
              'diningtable' : 11,
              'dog' : 12,
              'horse' : 13,
              'motorbike' : 14,
              'person' : 15,
              'pottedplant' : 16,
              'sheep' : 17,
              'sofa' : 18,
              'train' : 19,
              'tvmonitor' : 20,
                'cells' : 22}
color_scheme = {'background' : 0, 'cells' : 22}

colors = [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                  [145, 30, 180], [70, 240, 240],[240, 50, 230], [210, 245, 60], [250, 190, 212],
                  [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
                  [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
                  [255, 255, 255], [0, 0, 0], [0, 130, 255]]


categories = os.listdir(path_to_experiments)

def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    arguments
    x		-- width in pixels
    sx		-- width standard deviation

    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = np.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M


def parse_fixations(fixations):
    """Returns all relevant data from a list of fixation ending events

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']

    returns

    fix		-	a dict with three keys: 'x', 'y', and 'dur' (each contain
                a numpy array) for the x and y coordinates and duration of
                each fixation
    """

    # empty arrays to contain fixation coordinates
    fix = {'x': np.zeros(len(fixations)),
           'y': np.zeros(len(fixations)),
           'dur': np.zeros(len(fixations))}
    # get all fixation coordinates
    for fixnr in range(len(fixations)):
        stime, etime, dur, ex, ey = fixations[fixnr]
        fix['x'][fixnr] = ex
        fix['y'][fixnr] = ey
        fix['dur'][fixnr] = dur

    return fix

#if not pathlib.Path(os.path.join(path_to_save_trainings_dataset, experiment)).is_dir():
#    os.mkdir(os.path.join(path_to_save_trainings_dataset, experiment))

for categorie in categories:
    pickle_files = os.listdir(os.path.join(path_to_experiments, categorie))

    if not pathlib.Path(os.path.join(path_to_save_trainings_dataset, categorie)).is_dir():
        os.mkdir(os.path.join(path_to_save_trainings_dataset, categorie))
        os.mkdir(os.path.join(path_to_save_trainings_dataset, categorie, 'cropped'))
        os.mkdir(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped'))
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'only_gaze_data')).mkdir(parents=True,
                                                                                                    exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'only_fixations')).mkdir(parents=True,
                                                                                                    exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'gaze_data')).mkdir(parents=True,
                                                                                        exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'fixations')).mkdir(parents=True,
                                                                                     exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'masks')).mkdir(parents=True,
                                                                                         exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'original')).mkdir(parents=True,
                                                                                                exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'gaze_map')).mkdir(parents=True,
                                                                                                   exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'fixation_map')).mkdir(parents=True,
                                                                                                   exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'heatmap')).mkdir(parents=True,
                                                                                                       exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'only_gaze_data')).mkdir(parents=True,
                                                                                                         exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'only_fixations')).mkdir(parents=True,
                                                                                                         exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'gaze_data')).mkdir(parents=True,
                                                                                                    exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'fixations')).mkdir(parents=True,
                                                                                                    exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'masks')).mkdir(parents=True,
                                                                                                exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'original')).mkdir(parents=True,
                                                                                                    exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'gaze_map')).mkdir(parents=True,
                                                                                                       exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'fixation_map')).mkdir(parents=True,
                                                                                                       exist_ok=True)
        Path(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'heatmap')).mkdir(parents=True,
                                                                                                           exist_ok=True)


    for file in pickle_files:
        file_to_read = open(os.path.join(path_to_experiments, categorie, file), 'rb')
        annotation_dict = pickle.load(file_to_read)

        image_name = annotation_dict['image_name']
        annotations = annotation_dict['annotations']

        path_to_image = os.path.join(path_to_images, categorie, 'original', image_name)
        path_to_mask = os.path.join(path_to_images, categorie, 'masks', image_name)
        image = Image.open(path_to_image)
        gaze_image_visualization = image.copy()
        fixation_image_visualization = image.copy()
        only_gaze_image_visualization = image.copy()
        only_fixation_image_visualization = image.copy()
        gaze_image_draw_image = ImageDraw.Draw(gaze_image_visualization, 'RGBA')
        fixation_image_draw_image = ImageDraw.Draw(fixation_image_visualization, 'RGBA')
        only_gaze_image_draw_image = ImageDraw.Draw(only_gaze_image_visualization, 'RGBA')
        only_fixation_image_draw_image = ImageDraw.Draw(only_fixation_image_visualization, 'RGBA')
        gaze_image_map = np.zeros((image.size[1], image.size[0]))
        fixation_map = np.zeros((image.size[1], image.size[0]))

        mask_image = cv2.imread(path_to_mask)
        mask_image = mask_image[:,:,0]

        heatmap_image = cv2.imread(os.path.join(path_to_ready_dataset_heatmaps, categorie, 'heatmaps_based_on_gaze_data', image_name))

        contours = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]


        for c in contours:
            coordinates = []
            for cc in c:
                coordinates.append((cc[0][0], cc[0][1]))
            color_cat = colors[color_scheme[categorie]]
            if len(coordinates) >= 2:
                gaze_image_draw_image.polygon(coordinates, fill=(color_cat[0], color_cat[1], color_cat[2], 200), outline='black')
                fixation_image_draw_image.polygon(coordinates, fill=(color_cat[0], color_cat[1], color_cat[2], 200), outline='black')
                #gaze_image_draw_image.polygon(coordinates, fill=(255, 255, 255, 180),
                #                              outline='black')
                #fixation_image_draw_image.polygon(coordinates, fill=(255, 255, 255, 180),
                #                                  outline='black')

        # calculate bbox for cropped images
        bbox = get_bbox(mask_image, pad=50, zero_pad=True)
        gaze_image_draw_image_copy = np.asarray(gaze_image_visualization, dtype=np.uint8)
        cropped_mask_image = crop_from_bbox(gaze_image_draw_image_copy, bbox, zero_pad=False)
        cropped_mask_image = Image.fromarray(cropped_mask_image)

        gaze_image_visualization.save(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'masks', image_name))
        cropped_mask_image.save(
            os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'masks', image_name.split('.png')[0] + '_mask.png'))

        width = image.size[1]
        height = image.size[0]

        gaze_data = annotations['gaze data']
        gaze_data_list = []
        for gaze_data_point in gaze_data:
            gaze_data_list.append((int(statistics.mean(
                [float(gaze_data_point['right_gaze_point_on_display_area'][0]), float(gaze_data_point['left_gaze_point_on_display_area'][0])])), int(statistics.mean(
                [float(gaze_data_point['right_gaze_point_on_display_area'][1]), float(gaze_data_point['left_gaze_point_on_display_area'][1])]))))

        for point in gaze_data_list:
            if int(point[1]) < image.size[1] and int(point[0]) < image.size[0] and int(point[1]) >= 0 and int(point[0]) >= 0:
                gaze_image_map[int(point[1])][int(point[0])] = 255
                gaze_image_draw_image.ellipse([int(point[0])-2, int(point[1])-2, int(point[0])+2, int(point[1])+2], fill=(255, 0, 0, 255), outline='red')
                only_gaze_image_draw_image.ellipse(
                    [int(point[0]) - 2, int(point[1]) - 2, int(point[0]) + 2, int(point[1]) + 2], fill=(255, 0, 0, 255),
                    outline='red')

        fixations = annotations['fixations']
        fixations_list = [(int(i[4]), int(i[3])) for i in fixations]

        for point in fixations_list:
            if int(point[1]) < image.size[0] and int(point[0]) < image.size[1] and int(point[1]) >= 0 and int(point[0]) >= 0:
                fixation_map[int(point[0])][int(point[1])] = 255
                fixation_image_draw_image.ellipse([int(point[1])-2, int(point[0])-2, int(point[1])+2, int(point[0])+2], fill=(255, 0, 0, 255), outline='red')
                only_fixation_image_draw_image.ellipse(
                    [int(point[1]) - 2, int(point[0]) - 2, int(point[1]) + 2, int(point[0]) + 2], fill=(255, 0, 0, 255),
                    outline='red')

        gaze_image_visualization_copy = np.asarray(gaze_image_visualization, dtype=np.uint8)
        fixation_image_visualization_copy = np.asarray(fixation_image_visualization, dtype=np.uint8)
        only_gaze_image_visualization_copy = np.asarray(only_gaze_image_visualization, dtype=np.uint8)
        only_fixation_image_visualization_copy = np.asarray(only_fixation_image_visualization, dtype=np.uint8)
        heatmap_image_copy = np.asarray(heatmap_image, dtype=np.uint8)
        image_copy = np.asarray(image, dtype=np.uint8)

        cropped_gaze_image_visualization = crop_from_bbox(gaze_image_visualization_copy, bbox, zero_pad=False)
        cropped_fixation_image_visualization = crop_from_bbox(fixation_image_visualization_copy, bbox, zero_pad=False)
        cropped_only_gaze_image_visualization = crop_from_bbox(only_gaze_image_visualization_copy, bbox, zero_pad=False)
        cropped_only_fixation_image_visualization = crop_from_bbox(only_fixation_image_visualization_copy, bbox, zero_pad=False)
        cropped_image = crop_from_bbox(image_copy, bbox, zero_pad=False)
        cropped_gaze_image_map = crop_from_bbox(gaze_image_map, bbox, zero_pad=False)
        cropped_fixation_map = crop_from_bbox(fixation_map, bbox, zero_pad=False)
        cropped_heatmap_image = crop_from_bbox(heatmap_image_copy, bbox, zero_pad=False)

        cropped_gaze_image_visualization = Image.fromarray(cropped_gaze_image_visualization)
        cropped_fixation_image_visualization = Image.fromarray(cropped_fixation_image_visualization)
        cropped_only_gaze_image_visualization = Image.fromarray(cropped_only_gaze_image_visualization)
        cropped_only_fixation_image_visualization = Image.fromarray(cropped_only_fixation_image_visualization)
        cropped_image = Image.fromarray(cropped_image)
        cropped_heatmap_image = Image.fromarray(cv2.cvtColor(cropped_heatmap_image, cv2.COLOR_BGR2RGB))

        gaze_image_visualization.save(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'gaze_data', image_name))
        fixation_image_visualization.save(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'fixations', image_name))
        only_gaze_image_visualization.save(
            os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'only_gaze_data', image_name))
        only_fixation_image_visualization.save(
            os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'only_fixations', image_name))
        image.save(
            os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'original', image_name))
        cv2.imwrite(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'gaze_map', image_name), gaze_image_map)
        cv2.imwrite(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'fixation_map', image_name),
                    fixation_map)
        cv2.imwrite(os.path.join(path_to_save_trainings_dataset, categorie, 'not_cropped', 'heatmap', image_name),
                    heatmap_image)

        cropped_gaze_image_visualization.save(
            os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'gaze_data', image_name.split('.png')[0] + '_gaze_data.png'))
        cropped_fixation_image_visualization.save(
            os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'fixations', image_name.split('.png')[0] + '_fixations.png'))
        cropped_only_gaze_image_visualization.save(
            os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'only_gaze_data', image_name.split('.png')[0] + '_gaze_data.png'))
        cropped_only_fixation_image_visualization.save(
            os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'only_fixations', image_name.split('.png')[0] + '_fixations.png'))
        cropped_heatmap_image.save(
            os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'heatmap', image_name.split('.png')[0] + '_heatmap.png'))
        cropped_image.save(
            os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'original', image_name.split('.png')[0] + '.png'))
        cv2.imwrite(os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'gaze_map', image_name.split('.png')[0] + '_gaze_data.png'),
                    cropped_gaze_image_map)
        cv2.imwrite(os.path.join(path_to_save_trainings_dataset, categorie, 'cropped', 'fixation_map', image_name.split('.png')[0] + '_fixations.png'), cropped_fixation_map)
