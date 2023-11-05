## Run the following command to install the required packages
```
pip install -r requirements.txt
```

## Run the following command to generate samples for the cda spectra

```
python3 cda_convert.py -p <path to the lilbid spectra> -n <no of samples> -d <data points in cda spectra>
```

example:
```
python3 cda_convert.py -p lilbid_data -n 50 -d 640
```

The above command will generate 50 samples of cda spectra with 640 data points each. The samples will be stored in the folder "lilbid_data/coverted_spectra" in the current directory.