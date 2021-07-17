# FCNloca

This is a simple demo about the FCN location. The basic idea is the same as in the SR paper (Zhang et al., 2020). number of training samples is 250; the number of testing samples is 100. However, if you want to apply the code to your own problem, you need to change the network structure, data IO, and so on. The code is extreamly simple, and there are only 250 samples for training for the demo code.

You can run the codes by CPU or GPU on Linux or Windows. The waveform data is preprocessed as SEGY data, and each SEGY corresponds to an earthquake event in the catalog (testing_samples.txt and training_samples). 
The catalog format:


SEGY_file   latitude   Longtitude   depth          origin_time

event0.sgy      3971.63        -10802            5.918      2016-03-31 21:31:29.600

....

The usage of the code is simple:

To run traning: python fcn_train.py

To run testing: python fcn_test.py


The output files include:

(1) the network model: FCNloca.hdf5; the training log: FCNloca.log

(2) the predicted location results: test_xyz.txt; the predicted location image labels: test_predictedlabel.mat; the true image lables: test_true.mat.

You can open txt file to check the location results or compare the location results with the true locations in catalog (testing_samples.txt). The mat files could be loaded in Matlab to plot the location images. Please find the matlab script in the ./plot folder.

Reference:
Zhang, X., Zhang, J., Yuan, C. et al. Locating induced earthquakes with a network of seismic stations in Oklahoma via a deep learning method. Sci Rep 10, 1941 (2020). https://doi.org/10.1038/s41598-020-58908-5
