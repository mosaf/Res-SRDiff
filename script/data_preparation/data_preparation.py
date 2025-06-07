import SimpleITK as sitk
import numpy as np
import os, h5py, shutil
import shutil

from tqdm import tqdm

def downsample_image(image, factor, interpolator=sitk.sitkLinear):
    """sitk.sitkBSpline
    Downsamples the input image by the specified factor.

    The new spacing is computed as: new_spacing = original_spacing * factor
    and the new size is computed as: new_size = round(original_size / factor)

    Parameters:
        image (SimpleITK.Image): The input image.
        factor (float): Downsampling factor. For example, factor=2 doubles the spacing.
        interpolator: SimpleITK interpolator type (default is BSpline).

    Returns:
        SimpleITK.Image: The downsampled image.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_origin = image.GetOrigin()
    original_direction = image.GetDirection()

    # Compute new spacing and size
    new_spacing = [s * factor for s in original_spacing]
    new_size = [int(round(sz / factor)) for sz in original_size]

    downsampled = sitk.Resample(
        image1=image,
        size=new_size,
        transform=sitk.Transform(),  # Identity transform
        interpolator=interpolator,
        outputOrigin=original_origin,
        outputSpacing=new_spacing,
        outputDirection=original_direction,
        defaultPixelValue=0,
        outputPixelType=image.GetPixelID()
    )
    return downsampled


def resample_to_reference(image, reference, interpolator=sitk.sitkBSpline):
    """
    Resamples the given image onto the grid defined by the reference image.

    Parameters:
        image (SimpleITK.Image): The image to be resampled.
        reference (SimpleITK.Image): The reference image defining the desired grid.
        interpolator: SimpleITK interpolator type (default is BSpline).

    Returns:
        SimpleITK.Image: The resampled image.
    """

    resampled = sitk.Resample(
        image1=image,
        referenceImage=reference,
        transform=sitk.Transform(),  # Identity transform
        interpolator=interpolator,
        defaultPixelValue=0,
        outputPixelType=image.GetPixelID()
    )
    return resampled


def remove_files_in_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory '{directory_path}' does not exist.")
        return

    # Iterate through the items in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        # Check if the item is a file or directory and remove it
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def load_mri_image_mask(file_path):
    mask = sitk.ReadImage(os.path.join(file_path, 'inv2_brain_mask.nii.gz')) # used to chose only slices with brain tissue
    img = sitk.ReadImage(os.path.join(file_path, 't1map_brain.nii.gz'))
    print("Original image spacing {}".format(img.GetSpacing()))
    img_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(mask)

    return img, mask, img_array, mask_array
def main():
    # File paths
    input_file = '/path/to/Nifti_files/'
    out_file = "/path/to/save_data/"

    downscale_factor = 4  # For example, reduce the resolution by a factor of 4
    out_file = os.path.join(out_file, str(downscale_factor))
    os.makedirs(out_file, exist_ok=True)

    remove_files_in_directory(out_file)

    pts = os.listdir(input_file)

    jj = 1
    for pt in tqdm(pts):
        image, mask, img_array, mask_array = load_mri_image_mask(os.path.join(input_file, pt))

        downsampled = downsample_image(image, downscale_factor)

        resampled = resample_to_reference(downsampled, image)
        low_res_data = sitk.Mask(resampled, mask)
        low_res_data = sitk.GetArrayFromImage(low_res_data)
        index = np.unique(np.where(mask_array == 1)[0])
        for ii in index[40:-20]: # exclude slices with brain tissue
            with h5py.File(os.path.join(out_file, f'sample_{jj}.h5'), 'w') as h5_file:
                # Save each array as a separate dataset in the HDF5 file
                h5_file.create_dataset('high_resolution', data=img_array[ii].astype(np.int16), compression='gzip')
                h5_file.create_dataset('low_resolution', data=low_res_data[ii].astype(np.int16), compression='gzip')
                h5_file.create_dataset('mask', data=mask_array[ii].astype(np.int16), compression='gzip')

            jj = jj + 1

if __name__ == "__main__":
    main()