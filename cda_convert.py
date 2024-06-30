import argparse
import numpy as np
from scipy import signal, stats
import pandas as pd
import matplotlib.pyplot as plt
from detectPeaks import detectPeaks
from baseline_correction import baseline_correction
import random
import glob
import os


def calc_shift_parameter(p1=0.1, p2=0.9):
    # This function calculates the shift parameter based on probability
    # p1 and p2.
    # p1: is the probability of selecting the shift parameter to be in
    # the range of (-0.55, 0)
    # p2: is the probability of selecting the shift parameter to be in
    # the range of (-1,-0.55)
    switch = [0, 1]
    prob = [p1, p2]
    shift = 0
    number = np.random.choice(switch, 1, p=prob)[0]
    if number == 1:
        shift = random.uniform(-0.55, 0)
    elif number == 0:
        shift = random.uniform(-1, -0.55)
    return round(shift, 2)


def calc_sn_ratio_highest_peak():
    observation_x = np.linspace(5, 500, 1000)
    pdf_skewnorm = stats.expon.pdf(observation_x, 5, 100)
    pdf_skewnorm = pdf_skewnorm / sum(pdf_skewnorm)
    sn_ratio_highest_wrt_peak = np.random.choice(observation_x, 1, p=pdf_skewnorm)[0]
    return int(sn_ratio_highest_wrt_peak)

def gen_noise(metadata_file, data_points, min_noise_value_range=(0.075,0.125), max_noise_value_range=(0.3,0.7)):
    # The following function help us to generate noise which which
    # mimics the noise of CDA spectra. The default values were selected
    # by ananlyzing the minimum and maximum noise values found in CDA
    # spectra.
    # data_points: number of data points in the CDA spectra
    # min_noise_value_range: range from which the minimum noise value will
    # be selected
    # max_noise_value_range: range from which the maximum noise value will
    # be selected.

    # c1 is the minimum noise value
    c1 = random.uniform(min_noise_value_range[0], min_noise_value_range[1])
    # c2 is the maximum noise value
    c2 = random.uniform(max_noise_value_range[0], max_noise_value_range[1])

    # The noise generated for each data point will be in the range of c1 and c2
    generated_noise = np.random.uniform(low=c1, high=c2, size=(data_points,))

    # Once we have the noise we multiply it by a factor to generate CDA
    # spectrum with different noise variations
    noise_multiple = random.randint(2, 8)
    generated_noise = noise_multiple * generated_noise
    metadata_file.write("Distortion Parameters\n")
    metadata_file.write("noise multiple: " + str(noise_multiple) + "\n")
    return generated_noise


def distortion(metadata_file, c1_range=(0.4, 1), c2_range=(0.3, 0.6), c3_range=(100, 200)):
    # The following function generates a trend which can be added as a distortion
    # in the conversion of CDA spectra, the variable c1, c2 and c3 are the hyperparameters
    # that allows us to generate random trends which mimic the trend found in CDA spectra.
    # The default values for c1, c2 and c3 were selected by analyzing the CDA spectra.
    c1 = random.uniform(c1_range[0], c1_range[1])
    c2 = random.uniform(c2_range[0], c2_range[1])
    c3 = random.randint(c3_range[0], c3_range[1])
    generated_trend = np.arange(0, 2.035 * np.pi, 0.01)
    generated_trend = c1 * np.sin(c2 * generated_trend)
    generated_trend[0:c3] = 0
    metadata_file.write("c1: " + str(c1) + "\n")
    metadata_file.write("c2: " + str(c2) + "\n")
    metadata_file.write("c3: " + str(c3) + "\n")
    return generated_trend

def gen_plot(t, converted_cda, filename):
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.set_yscale('log')
    ax.set_title('Spectrum')
    ax.set_xlabel('Time (microseconds)')
    ax.set_ylabel('Amplitude (V)')
    ax.plot(t, converted_cda)
    plt.savefig(filename)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_dir_path', type=str, help="Input path of the cda data dir", required=True)
    parser.add_argument('-n', '--number_of_samples', type=int, help="No of samples to be generated for each spectra")
    parser.add_argument('-d', '--data_points', type=int, help="No of data points in the generated spectra")
    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    data_dir_path = args.data_dir_path
    number_of_samples = args.number_of_samples
    data_points = args.data_points
    file_pattern = '*.txt'
    files = glob.glob(f"{data_dir_path}/{file_pattern}")

    for file in files:
        file_base_name = os.path.basename(file)
        file_dir = os.path.dirname(file)
        file_name = os.path.splitext(file_base_name)[0]
        raw_data = np.loadtxt(file)
        coverted_data_dir = os.path.join(file_dir, "coverted_spectra")
        if not os.path.exists(coverted_data_dir):
            os.makedirs(coverted_data_dir)
        data = baseline_correction(raw_data)
        peaks, peak_indices, local_percentile = detectPeaks(data)
        file_metadata = open(os.path.join(coverted_data_dir, f"conversion_metadata.txt"), "w")
        for n in range(number_of_samples):
            file_metadata.write("{}_conversion_sample{}\n".format(file_name, n))
            stretch_parameter = 0.475
            shift_parameter = calc_shift_parameter()
            sn_ratio_highest_peak = calc_sn_ratio_highest_peak()
            noise = gen_noise(file_metadata, data_points)
            trend = distortion(file_metadata)
            m = []
            t = np.arange(data_points) * 0.01
            for i in t:
                m.append((((i - shift_parameter) / stretch_parameter) ** 2))

            cda_converted = np.zeros((data_points, 2))
            cda_converted[:, 0] = m

            for i in range(len(peaks)):
                idx = (np.abs(cda_converted[:, 0] - peaks[:, 0][i])).argmin()
                cda_converted[idx][1] = peaks[:, 1][i]

            # Normalised peaks in cda form
            cda_converted[:, 1] = cda_converted[:, 1] / sum(cda_converted[:, 1])

            # Convolving normalised peaks
            win = signal.windows.gaussian(30, 3)
            y_conv = signal.convolve(cda_converted[:, 1], win, mode='same') / sum(win)

            file_metadata.write("Noise: " + str(np.average(noise)) + "\n")

            # Calculating highest signal
            file_metadata.write("Highest peak: " + str(np.max(y_conv * sn_ratio_highest_peak)) + "\n")

            # Multiplication factor
            factor = sn_ratio_highest_peak * np.average(noise) / np.max(y_conv)

            y_conv = y_conv * factor
            y_conv_noise = y_conv + noise + trend
            y_conv_noise[np.where(y_conv_noise < 0.1)[0]] = 0
            file_metadata.write("S/N: " + str(np.max(y_conv) / np.average(noise)) + "\n")
            file_metadata.write("\n")

            conversion = pd.DataFrame()
            conversion["time"] = np.round(t, 2)
            conversion["m"] = np.round(m, 2)
            conversion["amp"] = np.round(y_conv_noise, 2)
            conversion.to_csv(os.path.join(coverted_data_dir, f"{file_name}_cda_coverted_sample{n}.csv"), index=False)
            gen_plot(t, y_conv_noise, os.path.join(coverted_data_dir, f"{file_name}_cda_coverted_sample{n}.pdf"))
        file_metadata.close()

if __name__ == "__main__":
    main()
