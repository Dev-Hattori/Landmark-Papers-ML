#################################################
"""
Utilities 
"""
#################################################


##############################################
"""
Kaggle Utilities
"""
##############################################


def import_kaggle(creds='{"username":"hattori404","key":"6ad76fc68e2cf37898a898090ab99f23"}'):
    """
    creds: Kaggle API token
    return: imported Kaggle.api
    """
    from pathlib import Path
    # expand an initial path component ~( tilde symbol) or ~user in the given path to user’s home directory.
    cred_path = Path('~/.kaggle/kaggle.json').expanduser()
    if not cred_path.exists():
        cred_path.parent.mkdir(exist_ok=True)
        cred_path.write_text(creds)
        cred_path.chmod(0o600)
    from kaggle import api
    return api


def ds_info_from_url(url):
    """
    Returns the dataset name if it belongs to a competition else the user and set name along with [0] being type
    type => 'competitions'  or 'datasets'
    """
    ds = url.split('.com/')[1].split('/')
    url = url.split('.com/')[0]  # Should be https://www.kaggle or just kaggle
    if (url == 'https://www.kaggle' or url == 'kaggle') and (ds[0] == 'competitions' or ds[0] == 'datasets'):
        return ds
    else:
        print("This is not a url for a kaggle dataset")


def check_url(url  # Dataset slug (ie "zillow/zecon")
              ):
    '''Check if dataset exists'''
    api = import_kaggle()
    ds_info = ds_info_from_url(url=url)
    if ds_info[0] == 'datasets':
        ds = api.dataset_list(user=ds_info[1], search=ds_info[2])
    elif ds_info[0] == "competitions":
        ds = api.competitions_list(search=ds_info[1])
    return ds


def get_data(url,  # To the dataset
             path,  # To store the dataset
             unzip=True,
             force='False'  # Stop if the path exists
             ):
    from pathlib import Path
    if not force:
        assert not Path(path).exists()

    api = import_kaggle()

    ds_info = ds_info_from_url(url)

    # Download the dataset zip file
    if ds_info[0] == 'datasets':  # This is a apart of datasets
        slug = str(check_url(url)[0])
        print(f"Downloading {slug}")
        api.dataset_download_files(slug, str(path))
    elif ds_info[0] == "competitions":  # Dataset of a competition
        slug = ds_info[1]
        print(f"Downloading {slug}")
        api.competition_download_files(str(slug), str(path))

    print("Download Complete")

    if unzip:
        print("Unzipping the files")
        zipped_file = Path(path)/f"{slug.split('/')[-1]}.zip"
        import zipfile
        with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
            zip_ref.extractall(Path(path))
        zipped_file.unlink()  # Delete the zipfile
        print("Files Successfully unzipped")


#############################################################
"""
Non-Kaggle Dataset Utilities
"""
#############################################################


def download_dataset_from_zipurl(zip_url,  # The url of the zip file
                                 path,  # Path where the dataset should be extracted to
                                 unzip=True  # Whether to unzip the zip file or not
                                 ):
    import requests
    import zipfile
    import io
    from clint.textui import progress
    import os

    os.mkdir(path)
    r = requests.get(zip_url, stream=True)
    with open(path/'dataset.zip', 'wb') as fd:
        total_length = int(r.headers.get('content-length'))

        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
            fd.write(chunk)

    if unzip:
        extract_zip(path/'dataset.zip', path)


def extract_zip(from_path, to_path):
    import zipfile
    z = zipfile.ZipFile(from_path)
    z.extractall(to_path)
