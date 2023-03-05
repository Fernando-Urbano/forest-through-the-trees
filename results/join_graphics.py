from os import listdir, remove
from os.path import isfile, join
from PIL import Image
import re


def group_images_by_name(folder_path="results/graphics"):
    # Get all the PNG files in the folder
    files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith('.png')]
    # Group the images by their name
    groups = {}
    for file_name in files:
        # Get the group name (i.e. the part of the name before "_leaf")
        group_name = file_name.split('_leaf')[0] + "_leaf"
        # Add the file to the group
        if group_name in groups:
            groups[group_name].append(file_name)
        else:
            groups[group_name] = [file_name]
    # Join the images vertically for each group
    for group_name in groups.keys():
        # Get the files for this group
        group_files = groups[group_name]
        # Filter the files to include only the ones we want
        sharpe_file = None
        tvalue_file = None
        rsquared_file = None
        for file_name in group_files:
            if 'sharpe' in file_name:
                sharpe_file = file_name
            elif 'tvalue' in file_name:
                tvalue_file = file_name
            elif 'rsquared' in file_name:
                rsquared_file = file_name
        # If we have all three files, join them vertically and save the result
        if sharpe_file and tvalue_file and rsquared_file:
            images = [
                Image.open(join(folder_path, sharpe_file)),
                Image.open(join(folder_path, tvalue_file)),
                Image.open(join(folder_path, rsquared_file))
            ]
            result = Image.new('RGBA', (images[0].width, sum([im.height for im in images])))
            y_offset = 0
            for im in images:
                result.paste(im, (0, y_offset))
                y_offset += im.height
            result.save(join(folder_path, f'{group_name}_results.png'))
            for file_name in group_files:
                if not bool(re.search("_results[.]png", file_name)):
                    remove(join(folder_path, file_name))


group_images_by_name()