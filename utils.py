from one.params import get_cache_dir
from one.api import ONE
from one.remote.aws import get_s3_from_alyx
from pathlib import Path

one = ONE(base_url='https://openalyx.internationalbrainlab.org/')

S3_DATA_PATH = 'paper_reproducible_ephys/mtnn_Q2_2024'

def save_data_path(figure=None):
    try:
        import reproducible_ephys_paths
        defined_paths = dir(reproducible_ephys_paths)
        if 'DATA_PATH' in defined_paths:
            data_path = Path(reproducible_ephys_paths.DATA_PATH)
        else:
            data_path = Path(get_cache_dir()).joinpath('paper_repro_ephys_data')

    except ModuleNotFoundError:
        data_path = Path(get_cache_dir()).joinpath('paper_repro_ephys_data')

    if figure is not None:
        data_path = data_path.joinpath(str(figure))

    data_path.mkdir(exist_ok=True, parents=True)

    return data_path

data_path = save_data_path(figure='fig_mtnn')

def download_aws(folder):
    s3, bucket_name = get_s3_from_alyx(one.alyx)
    bucket = s3.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=f'{S3_DATA_PATH}/{folder}'):
        download_path = data_path.joinpath(Path(obj.key).relative_to(S3_DATA_PATH))
        download_path.parent.mkdir(exist_ok=True, parents=True)
        bucket.download_file(obj.key, str(download_path))

def download_glm_hmm():
    if data_path.joinpath('glm_hmm').exists():
        return
    print('downloading glm_hmm data')
    download_aws('glm_hmm')

# ephys_eid from Matt - the invalid ones removed
val_eid = [
    'db4df448-e449-4a6f-a0e7-288711e7a75a',  # Berkeley
    'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',  # Berkeley
    '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',  # Berkeley
    'e535fb62-e245-4a48-b119-88ce62a6fe67',  # Berkeley
    '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',  # Berkeley
    'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0',  # Berkeley
    '30c4e2ab-dffc-499d-aae4-e51d6b3218c2',  # CCU
    'd0ea3148-948d-4817-94f8-dcaf2342bbbe',  # CCU
    'a4a74102-2af5-45dc-9e41-ef7f5aed88be',  # CCU
    '746d1902-fa59-4cab-b0aa-013be36060d5',  # CCU
    '88224abb-5746-431f-9c17-17d7ef806e6a',  # CCU
    '0802ced5-33a3-405e-8336-b65ebc5cb07c',  # CCU
    'ee40aece-cffd-4edb-a4b6-155f158c666a',  # CCU
    'c7248e09-8c0d-40f2-9eb4-700a8973d8c8',  # CCU
    '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',  # CCU
    'dda5fc59-f09a-4256-9fb5-66c67667a466',  # CSHL(C)
    '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',  # CSHL(C)
    'f312aaec-3b6f-44b3-86b4-3a0c119c0438',  # CSHL(C)
    '4b00df29-3769-43be-bb40-128b1cba6d35',  # CSHL(C)
    'ecb5520d-1358-434c-95ec-93687ecd1396',  # CSHL(C)
    '51e53aff-1d5d-4182-a684-aba783d50ae5',  # NYU
    'f140a2ec-fd49-4814-994a-fe3476f14e66',  # NYU
    'a8a8af78-16de-4841-ab07-fde4b5281a03',  # NYU
    '61e11a11-ab65-48fb-ae08-3cb80662e5d6',  # NYU
    '73918ae1-e4fd-4c18-b132-00cb555b1ad2',  # Princeton
    'd9f0c293-df4c-410a-846d-842e47c6b502',  # Princeton
    'dac3a4c1-b666-4de0-87e8-8c514483cacf',  # SWC(H)
    '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',  # SWC(H)
    '56b57c38-2699-4091-90a8-aba35103155e',  # SWC(M)
    '3638d102-e8b6-4230-8742-e548cd87a949',  # SWC(M)
    '7cb81727-2097-4b52-b480-c89867b5b34c',  # SWC(M)
    '781b35fd-e1f0-4d14-b2bb-95b7263082bb',  # UCL
    '3f859b5c-e73a-4044-b49e-34bb81e96715',  # UCL
    'b22f694e-4a34-4142-ab9d-2556c3487086',  # UCL
    '0a018f12-ee06-4b11-97aa-bbbff5448e9f',  # UCL
    'aad23144-0e52-4eac-80c5-c4ee2decb198',  # UCL
    'e45481fa-be22-4365-972c-e7404ed8ab5a',  # UCL
    'd04feec7-d0b7-4f35-af89-0232dd975bf0',  # UCL
    '1b715600-0cbc-442c-bd00-5b0ac2865de1',  # UCL
    'c7bf2d49-4937-4597-b307-9f39cb1c7b16',  # UCL
    '8928f98a-b411-497e-aa4b-aa752434686d',  # UCL
    'ebce500b-c530-47de-8cb1-963c552703ea',  # UCLA
    'dc962048-89bb-4e6a-96a9-b062a2be1426',  # UCLA
    '6899a67d-2e53-4215-a52a-c7021b5da5d4',  # UCLA
    '15b69921-d471-4ded-8814-2adad954bcd8',  # UCLA
    '824cf03d-4012-4ab1-b499-c83a92c5589e',  # UCLA
    '3bcb81b4-d9ca-4fc9-a1cd-353a966239ca',  # UW
    'f115196e-8dfe-4d2a-8af3-8206d93c1729',  # UW
    '9b528ad0-4599-4a55-9148-96cc1d93fb24',  # UW
    '3e6a97d3-3991-49e2-b346-6948cb4580fb',  # UW
]