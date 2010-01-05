import numpy

class Dist:

    def __init__(self, matrix, mode="dtw", fast=False, radius=20, pu="CPU"):
        '''
            <calls others modules for the actual calculation, checks if CUDA
             is present an organize computations>

            Input Parameters:
                matrix: a matrix containing time series, each row is a
                timeseries and each column a point of the time series.

                mode: method by which the distance between series is
                      calculated
                  "dtw" for normal DTW,
                  "ddtw" for the derivative DTW,
                  "euclidean" for euclidean distance,
                  "pearson", for pearson distance

                fast: specifies whether to run the FastDTW or not, radius
                      parameter indicates the accuracy of the calculation,
                      the higher the value, the more accurate the result.
                      It applies only to the DTW algorithms.
                      FastDTWs are not implemented on GPU.

                radius: see 'fast'. It is suggested to not use more than
                        100 for this value ignored if fast is flase.

            Returns:

                None
        '''
        if pu == "GPU":
            try:
                import dtw_gpu
            except ImportError:
                print "No suitable hardware! Doing DTW on CPU..."
                self.pu = "CPU"
            else:
                self.gpu = dtw_gpu._DTW_(self.matrix)
        else:
            import dtw_cpu
            self.dtw_cpu = dtw_cpu
        self.pu = pu
        self.matrix = matrix
        self.mode = mode
        self.fast = fast
        self.radius = radius
        self.derivative = False
        self.euclidean = False
        if self.mode == "ddtw":
            self.derivative = True
        elif self.mode == "euclidean":
            self.euclidean = True

    def compute(self, li):
        '''
            <Does the actual calculations.>

            Input Parameters:

                li: list of tuples containg indices of couples of time
                    series between which distace has to be calculated.
                    Indices refears to matrix passed at the init func.

            Returns:

                returns a numpy array containing distance values for each
                couple passed. Result
                are in the order of input.
        '''
        if self.pu == "GPU":
            res = self.gpu.compute_dtw(li)
        else:
            res = numpy.array([])
            for qui in li:
                tmp = self.dtw_cpu.compute_dtw(self.matrix[qui[0]],
                                          self.matrix[qui[1]],
                                          self.euclidean,
                                          False,
                                          self.derivative,
                                          self.fast,
                                          self.radius)
                res = numpy.append(res, numpy.array([tmp]))
        return res
