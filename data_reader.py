# Author: Kelvin Chng
# (c) 2016
# San Jose State University

import numpy as np
import random
import time
import sys

class insert_file_info :
    
    def __init__(self, full_file_path, filenumber, batch_size = 50, 
        use_random_seed = False, include_validation_data = False, 
        load_test_data_only = False) :
        """ full_file_path : full file path of the shuffled data
            filenumber     : An array of file number """
        self.filename         = full_file_path.rsplit('\\', 1)[-1]
        self.filename         = self.filename.rsplit('/', 1)[-1]
        self.filenumber       = filenumber
        self.full_file_path   = full_file_path
        self.include_validation_data = include_validation_data
        self.nrows            = 0
        self.ncols            = 0
        self.nfile            = len(filenumber)
        self.batch_size       = batch_size
        self.current_index    = 0
        self.load_test_data_only = load_test_data_only
        if self.load_test_data_only :
            self.include_validation_data = False
        self.delimiter        = [1 for i in np.arange(self.ncols)]

    class DataSet(object) :
        file_info = None
    
        def __init__(self, images, labels, temps, signs, nrows, nfile_train, 
                     nfile_test, nfile_val, full_file_path, data_type) :
            #self.file_into = insert_file_info()
        
            #super(DataSet,self).__init__()
            #self.insert_file_info = insert_file_info

            self._epochs_completed = 0
            self._file_index = 1
            self._images = images
            self._index_in_datafile = 0
            self._index_in_epoch = 0
            self._labels = labels
            self._ndata = 0
            self._temps = temps
            self._signs = signs
            self.batch_size = 0
            self.data_type = data_type
            self.full_file_path = full_file_path
            self.nrows = nrows
            self.shuffle_index_dose = np.arange(0,self.nrows,1)
             
            if self.data_type == 'train' :
                self.start_file_index   = 1
                self.end_file_index     = nfile_train
                self._ndata             = nfile_train*self.nrows
                self.convert_to_one_hot = True
                self.shuffle_index = np.arange(0,self._ndata,1)
            elif self.data_type == 'test' :
                self.start_file_index   = nfile_train + 1
                self.end_file_index     = nfile_train + nfile_test
                self._ndata             = nfile_test*self.nrows
                self.convert_to_one_hot = True
                self.shuffle_index = np.arange(0,self._ndata,1)
            elif self.data_type == 'validation' :
                self.start_file_index   = nfile_train + nfile_test + 1
                self.end_file_index     = nfile_train + nfile_test + nfile_val
                self._ndata             = nfile_val*self.nrows
                self.convert_to_one_hot = False
                self.shuffle_index = np.arange(0,self._ndata,1)

        #@staticmethod
        #def feed_self(self, batch_size, nrows) :
        #    self.batch_size = batch_size
        #    self.nrows      = nrows
            #print self.batch_size, self.nrows

        @property
        def images(self):
            return self._images
        
        @property
        def labels(self):
            return self._labels
   
        @property
        def temps(self):
            return self._temps

        @property
        def signs(self):
            return self._signs

        @property
        def ndata(self):
            return self._ndata
    
        @property
        def epochs_completed(self):
            return self._epochs_completed
            
        def next_batch(self, batch_size = 50) :
            
            start = self._index_in_epoch
            if ( self._epochs_completed == 0 ) and ( start == 0 ) :
                self.batch_size = batch_size
                while np.modf(float(self._ndata)/self.batch_size)[0] > 0.0 :
                     print('Warning! Number of data/ batch size must be an integer.')
                     print('number of data: %d' % self._ndata)
                     print('batch size: %d'     % self.batch_size)
                     self.batch_size = int(input('Input new batch size: '))
                print('batch size : %d'    % self.batch_size)
                print('number of data: %d' % self._ndata)

            self._index_in_epoch += self.batch_size
            if self._index_in_epoch > self._ndata :
                # Number of training epochs completed
                self._epochs_completed += 1
                # Shuffle data
                random.shuffle(self.shuffle_index)
                self._images = self._images[self.shuffle_index]
                self._labels = self._labels[self.shuffle_index]
                # Reinitialize conunter
                start = 0
                self._index_in_epoch = self.batch_size
                assert self.batch_size <= self._ndata
            else :
                random.shuffle(self.shuffle_index)
                self._images = self._images[self.shuffle_index]
                self._labels = self._labels[self.shuffle_index]
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

        def next_dose(self, batch_size = 50) :

            def convert_to_one_hot( label ) :
                label_one_hot = np.zeros((len(label),2))
                for i in range(len(label)) :
                    label_one_hot[i,label[i]] = 1
                return label_one_hot

            start = self._index_in_datafile
            if ( self._file_index == self.start_file_index ) and ( start == 0 ) :
                self.batch_size = batch_size
                while np.modf(float(self.nrows)/self.batch_size)[0] > 0.0 :
                     print('Warning! Number of data per file/ dose size must be an integer.')
                     print('number of data per file: %d' % self.nrows)
                     print('dose size: %d'               % self.batch_size)
                     self.batch_size = int(input('Input new dose size: '))
                print('dose size : %d'    % self.batch_size)
                print('number of data: %d' % self._ndata)
                # Read in one file at a time
                data = np.genfromtxt(self.full_file_path%(self._file_index) ,dtype=int,
                       skip_header=0, skip_footer=0)
                self._images = data[:,:-1].astype('int')
                labels = data[:,-1:].astype('int')
                if self.convert_to_one_hot :
                    self._labels = convert_to_one_hot(labels)

            self._index_in_datafile += self.batch_size
            if self._index_in_datafile > self.nrows :
                self._file_index += 1
                start = 0
                self._index_in_datafile = self.batch_size
                assert self.batch_size <= self.nrows
                # Read in one file at a time
                data = np.genfromtxt(self.full_file_path%(self._file_index) ,dtype=int,
                       skip_header=0, skip_footer=0)
                self._images = data[:,:-1].astype('int')
                labels = data[:,-1:].astype('int')
                if self.convert_to_one_hot :
                    self._labels = convert_to_one_hot(labels)
                # Shufle data
                random.shuffle(self.shuffle_index_dose)
                self._images = self._images[self.shuffle_index_dose]
                self._labels = self._labels[self.shuffle_index_dose]

            if self._file_index > self.end_file_index :
                # Number of training epochs completed
                self._epochs_completed += 1
                self._file_index = self.start_file_index
                # Reinitialize conunter
                start = 0
                self._index_in_datafile = self.batch_size

            end = self._index_in_datafile

            return self._images[start:end], self._labels[start:end]

        def next_dose_old(self, batch_size = 50) :

            def convert_to_one_hot( label ) :
                label_one_hot = np.zeros((len(label),2))
                for i in range(len(label)) :
                    label_one_hot[i,label[i]] = 1
                return label_one_hot

            start = self._index_in_datafile 
            if ( self._file_index == self.start_file_index ) and ( start == 0 ) :
                self.batch_size = batch_size
                while np.modf(float(self.nrows)/self.batch_size)[0] > 0.0 :
                     print('Warning! Number of data per file/ dose size must be an integer.')
                     print('number of data per file: %d' % self.nrows)
                     print('dose size: %d'               % self.batch_size)
                     self.batch_size = int(input('Input new dose size: '))
                print('dose size : %d'    % self.batch_size)
                print('number of data: %d' % self._ndata)
                self.shuffle_index_dose_old = np.arange(0,self.batch_size,1)

            self._index_in_datafile += self.batch_size
            if self._index_in_datafile > self.nrows :
                self._file_index += 1
                start = 0
                self._index_in_datafile = self.batch_size
                assert self.batch_size <= self.nrows

            if self._file_index > self.end_file_index :
                # Number of training epochs completed
                self._epochs_completed += 1
                self._file_index = self.start_file_index
                # Reinitialize conunter
                start = 0
                self._index_in_datafile = self.batch_size

            end = self._index_in_datafile

            # Read in small dosage of data
            data = np.genfromtxt(self.full_file_path%(self._file_index) ,dtype=int,
                   skip_header=start, skip_footer=self.nrows-end)
            self._images = data[:,:-1].astype('int')
            labels = data[:,-1:].astype('int')
            if self.convert_to_one_hot :
                self._labels = convert_to_one_hot(labels)
            # Shufle data
            random.shuffle(self.shuffle_index_dose_old)
            self._images = self._images[self.shuffle_index_dose_old]
            self._labels = self._labels[self.shuffle_index_dose_old]

            return self._images, self._labels

    def categorize_data(self, convert_test_labels_to_one_hot = True, make_spin_down_negative = False) :
        class DataSets(object):
            pass
        data_sets = DataSets()
        
        def convert_to_one_hot( label ) :
            label_one_hot = np.zeros((len(label),2))
            for i in range(len(label)) :
                label_one_hot[i,label[i]] = 1
            return label_one_hot

        def reindex_data( in_data, L=200 ) :

            nrows, ncols = data_shape = np.shape(in_data)
            n_x = int(round((float(ncols)/L)**(1/3.)))
            index = range(ncols)
            new_index = np.zeros(ncols)
            count=0

            for j in range(L) :
                 for i in range(n_x**3) :
                    new_index[count] = index[j+i*L]
                    count+=1
            output_data = np.zeros(np.shape(in_data))
            for i in range(ncols) :
                output_data[:,int(new_index[i])] = in_data[:,i]

            return output_data
 
        data = np.loadtxt(self.full_file_path%1)
        self.nrows, self.ncols = np.shape(data)
        self.nrows, self.ncols = int(self.nrows), int(self.ncols)
        
        if np.modf(float(self.nrows)/self.batch_size)[0] > 0.0 :
            self.batch_size = int(float(self.nrows)/20)    
           
        if self.include_validation_data :
           # Use 10% of the data each for testing and validating, the remaining for
           # training    
           nfile_train = int(self.nfile*.8)
           nfile_test  = int(self.nfile*.1)
           nfile_val   = nfile_test
        else :
           # Use 15% of the data for testing, the remaining for training
           nfile_train = int(self.nfile*.5)
           nfile_test  = int(self.nfile*.5)
           nfile_val   = 0
    
        n_data_check = self.nfile - ( nfile_train + nfile_test + nfile_val )
        if n_data_check > 0 :
            nfile_train += n_data_check
        elif n_data_check < 0 :
            nfile_train -= n_data_check
   
        start_time = time.time()
       
        if not(self.load_test_data_only) :

            TRAIN_DATA = np.zeros((nfile_train*self.nrows,self.ncols))
            #train_images = np.zeros((nfile_train*self.nrows,self.ncols-1))
            #train_labels = np.zeros((nfile_train*self.nrows,1))
            print('Loading %d/%d files for training data...' % (nfile_train,self.nfile))
            for i in range(nfile_train) :
                print('%.1fs. Loading file %d.' % (time.time()-start_time, i+1))
                TRAIN_DATA[i*self.nrows:(i+1)*self.nrows,:] = np.loadtxt(self.full_file_path%(i+1))
            train_images = TRAIN_DATA[:,1:-1].astype('int')
            if make_spin_down_negative :
                train_images[train_images==0] = -1
            train_labels = TRAIN_DATA[:,-1].astype('int')
            train_labels = convert_to_one_hot(train_labels)
            train_temps = []
            train_signs = []

        print('Loading %d/%d files for test data...' % (nfile_test,self.nfile))
        TEST_DATA = np.zeros((nfile_test*self.nrows,self.ncols))
        #test_images = np.zeros((nfile_test*self.nrows,self.ncols-1))
        #test_labels = np.zeros((nfile_test*self.nrows,1))
        for i in range(nfile_test) :
            print('%.1fs. Loading file %d.' % (time.time()-start_time, i+1))
            TEST_DATA[i*self.nrows:(i+1)*self.nrows,:] = np.loadtxt(self.full_file_path%(i+1+nfile_train))
        test_images = TEST_DATA[:,1:-1].astype('int')
        if make_spin_down_negative :
            test_images[test_images==0] = -1
        test_labels = TEST_DATA[:,-1].astype('int')
        if convert_test_labels_to_one_hot :
            test_labels = convert_to_one_hot(test_labels)
        test_temps  = TEST_DATA[:,0] 
        test_signs  = []

        if self.include_validation_data :
            print('Loading %d/%d files for validation data...' % (nfile_val,self.nfile))
            VALIDATION_DATA = np.zeros((nfile_val*self.nrows,self.ncols))
            #validation_images = np.zeros((nfile_val*self.nrows,self.ncols-1))
            #validation_labels = np.zeros((nfile_val*self.nrows,1))
            for i in range(nfile_test) :
                print('%.1fs. Loading file %d.' % (time.time()-start_time, i+1))
                VALIDATION_DATA[i*self.nrows:(i+1)*self.nrows,:] = np.loadtxt(self.full_file_path%(i+1+nfile_train+nfile_test))
            validation_images = VALIDATION_DATA[:,:-2].astype('int')
            if make_spin_down_negative :
                validation_images[validation_images==0] = -1
            validation_labels = VALIDATION_DATA[:,-2].astype('int')
            validation_temps  = VALIDATION_DATA[:,-1].astype('int')
            validation_signs  = []

        if not(self.load_test_data_only) :
            data_sets.train      = insert_file_info.DataSet(train_images, train_labels,
                                   train_temps, train_signs, self.nrows, nfile_train, 
                                   nfile_test, nfile_val, self.full_file_path, 
                                   data_type = 'train')
        data_sets.test           = insert_file_info.DataSet(test_images, test_labels,
                                   test_temps, test_signs, self.nrows, nfile_train,
                                   nfile_test, nfile_val, self.full_file_path, data_type = 'test')
        if self.include_validation_data :
            data_sets.validation = insert_file_info.DataSet(validation_images,
                                   validation_labels, validation_temps, validation_signs, 
                                   self.nrows, nfile_train, nfile_test, nfile_val, 
                                   self.full_file_path, data_type = 'validation')

        return data_sets
 
    def categorize_dose_of_data(self) :
        class DataSets(object):
            pass
        data_sets = DataSets()

        data = np.loadtxt(self.full_file_path%1)
        self.nrows, self.ncols = np.shape(data)
        self.nrows, self.ncols = int(self.nrows), int(self.ncols)

        if np.modf(float(self.nrows)/self.batch_size)[0] > 0.0 :
            self.batch_size = int(float(self.nrows)/20)

        if self.include_validation_data :
           # Use 10% of the data each for testing and validating, the remaining for
           # training    
           nfile_train = int(self.nfile*.8)
           nfile_test  = int(self.nfile*.1)
           nfile_val   = nfile_test
        else :
           # Use 10% of the data each for testing, the remaining for training    
           nfile_train = int(self.nfile*.5)
           nfile_test  = int(self.nfile*.5)
           nfile_val   = 0

        n_data_check = self.nfile - ( nfile_train + nfile_test + nfile_val )
        if n_data_check > 0 :
            nfile_train += n_data_check
        elif n_data_check < 0 :
            nfile_train -= n_data_check
        
        if not(self.load_test_data_only) :
 
            train_images = np.array([]).astype('int')
            train_labels = np.array([]).astype('int')
            train_temps = []
            train_signs = []

        start_time = time.time()

        print('Loading %d/%d files for test data...' % (nfile_test,self.nfile))
        TEST_DATA = np.zeros((nfile_test*self.nrows,self.ncols))
        test_images = np.zeros((nfile_test*self.nrows,self.ncols-1))
        test_labels = np.zeros((nfile_test*self.nrows,1))
        for i in range(nfile_test) :
            print('%.1fs. Loading file %d.' % (time.time()-start_time, i+1))
            TEST_DATA[i*self.nrows:(i+1)*self.nrows,:] = np.loadtxt(self.full_file_path%(i+1+nfile_train))
        test_images = TEST_DATA[:,:-2].astype('int')
        test_labels = TEST_DATA[:,-2].astype('int')
        if convert_test_labels_to_one_hot :
            test_labels = convert_to_one_hot(test_labels)
        test_temps  = TEST_DATA[:,-1].astype('int')
        test_signs  = []

        if self.include_validation_data :
            print('Loading %d/%d files for validation data...' % (nfile_val,self.nfile))
            VALIDATION_DATA = np.zeros((nfile_val*self.nrows,self.ncols))
            validation_images = np.zeros((nfile_val*self.nrows,self.ncols-1))
            validation_labels = np.zeros((nfile_val*self.nrows,1))
            for i in range(nfile_test) :
                print('%.1fs. Loading file %d.' % (time.time()-start_time, i+1))
                VALIDATION_DATA[i*self.nrows:(i+1)*self.nrows,:] = np.loadtxt(self.full_file_path%(i+1+nfile_train+nfile_test))
            validation_images = VALIDATION_DATA[:,:-2].astype('int')
            validation_labels = VALIDATION_DATA[:,-2].astype('int')
            validation_temps  = VALIDATION_DATA[:,-1].astype('int')
            validation_signs  = []

        #test_images = np.array([]).astype('int')
        #test_labels = np.array([]).astype('int') 
        #test_temps  = np.array([]).astype('int')

        #if self.include_validation_data :
        #    validation_images = np.array([]).astype('int')
        #    validation_labels = np.array([]).astype('int')
        #    validation_temps  = np.array([]).astype('int')

        if not(self.load_test_data_only) :
            data_sets.train      = insert_file_info.DataSet(train_images, train_labels,
                                   train_temps, train_signs, self.nrows, nfile_train,
                                   nfile_test, nfile_val, self.full_file_path,
                                   data_type = 'train')
        data_sets.test           = insert_file_info.DataSet(test_images, test_labels,
                                   test_temps, test_signs, self.nrows, nfile_train,
                                   nfile_test, nfile_val, self.full_file_path,
                                    data_type = 'test')
        if self.include_validation_data :
            data_sets.validation = insert_file_info.DataSet(validation_images,
                                   validation_labels, validation_temps, validation_signs, 
                                   self.nrows,  nfile_train, nfile_test, nfile_val, 
                                   self.full_file_path, data_type = 'validation')

        return data_sets

    def load_classification_data(self, nrows = 1000, ncols=12800, SkipHeader = 0, load_ndata_per_file = 1000, include_sign=False, make_spin_down_negative = False) :
        class DataSets(object):
            pass
        data_sets = DataSets()

        def reindex_data( in_data, L=200 ) :

            nrows, ncols = data_shape = np.shape(in_data)
            n_x = int(round((float(ncols)/L)**(1/3.)))
            index = range(ncols)
            new_index = np.zeros(ncols)
            count=0

            for j in range(L) :
                 for i in range(n_x**3) :
                    new_index[count] = index[j+i*L]
                    count+=1
            output_data = np.zeros(np.shape(in_data))
            for i in range(ncols) :
                output_data[:,int(new_index[i])] = in_data[:,i]

            return output_data

        start_time = time.time()

        self.ncols            = ncols
        self.nrows            = nrows
        self.delimiter        = [1 for i in xrange(self.ncols)]

        #if SkipHeader == 0 :
        #    load_ndata_per_file = self.nrows
        SkipFooter = self.nrows - SkipHeader - load_ndata_per_file

        while load_ndata_per_file > self.nrows :
            print('Number of classification data used per temperature must be smaller than number of data per temnperature.')
            print('Number of data per temnperature         : %d' % self.nrows)
            print('Classification data used per temperature: %d' % load_ndata_per_file)
            load_ndata_per_file = input('Input new classification data used per temperature: ')

        classification_images = np.zeros((self.nfile*load_ndata_per_file,self.ncols))
        print('Loading %d files for classfication data...' % (self.nfile))
        for i in range(self.nfile) :
            print('%.1fs. Loading file %d.' % (time.time()-start_time, i+1))
            classification_images[i*load_ndata_per_file:(i+1)*load_ndata_per_file,:] = np.genfromtxt(self.full_file_path%self.filenumber[i], dtype = int, delimiter=self.delimiter, skip_header=SkipHeader, skip_footer=SkipFooter)
        classification_images = classification_images.astype('int')
        if make_spin_down_negative :
            classification_images[classification_images==0] = -1
        classification_labels = []
        classification_temps  = []

        if include_sign :
            classification_signs = np.zeros(self.ncols)
            for i in range(self.nfile) :
                classification_signs[i*load_ndata_per_file:(i+1)*load_ndata_per_file] = np.loadtxt(self.full_file_path%self.filenumber[i])[SkipHeader:(self.nrows-SkipFooter),-1]
        else :
            classification_signs = []

        data_sets.classification = insert_file_info.DataSet(classification_images, classification_labels,
                                   classification_temps, classification_signs, 0, 0, 0, 0, self.full_file_path,
                                   data_type='classification')

        return data_sets
