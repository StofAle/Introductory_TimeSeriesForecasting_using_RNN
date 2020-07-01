#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 06/20/2020
Author: Alexander Stoffers

"""

# load modules
import tensorflow as tf

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import numpy as np
import pandas as pd
import time

class TimeHistory(tf.keras.callbacks.Callback):
    """ class to record time it takes for each epoch in model fit
    """
    
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class Kind_of_Blue(object):
    """ some infos here
    
    methods:
        
    """
    
    ### class attributes here
    TRAIN_SPLIT_RATIO = 0.7
    
    def __init__(self):
        
        # raw data
        self._dataset = None
        self._df = None
        
        # data fitting details
        self._selected_features = None
        self._train_split = None
        
        # training and validation data
        self._train_data = None
        self._val_data = None
        self._num_samples = None  # number of training samples
        
        # target
        self._target = None  # is assigned in generate_train_and_val_data()
        
        # model 
        self._models = {}  # dictionary containing key=model name, value=model
        # self._model = None
        # self._model_type = None
        
        # model fit details
        self._histories = {}
        self._time_callbacks = {}
        
        
        
    @property
    def df(self):
        return self._df


    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        self._df = df
        return None
    
    
    @df.deleter
    def df(self) -> pd.DataFrame:
        del self._df
        
    
    def initialize_dataset(self) -> None:
        """ initialize dataset to be used in modelling approach
        
        """
        
        if isinstance(self._df, pd.DataFrame):
            features = self._selected_features
            
            if features[0]:
                self._dataset = self._df[features].values
                
                self._train_split = int(len(self._df) * Kind_of_Blue.TRAIN_SPLIT_RATIO)
            else:
                print('please select feature for dataset')
        else:
            print('please load dataframe')

        return None
    
    
    def standardize_data(self) -> None:
        """
        

        Returns
        -------
        None
            DESCRIPTION.

        """
        # take training data only        
        data_mean = self._dataset[:self._train_split].mean(axis=0)  
        data_std = self._dataset[:self._train_split].std(axis=0)
        
        # standardize data
        self._dataset = (self._dataset-data_mean)/data_std  
        # note: dataset now np.array, no longer pd.DataFrame
        
        return None
            
    
    def slice_time_series_data(self, target: np.ndarray
                               , start_index: int, end_index: int
                               , history_size: int , target_size: int
                               , step: int=1
                               ) -> [np.ndarray, np.ndarray]:
        """ slice time series data
        

        Parameters
        ----------
        dataset : np.ndarray
            input dataset.
        target : np.ndarray
            to-be-predicted (target) values.
        start_index : int
            cut-off start index.
        end_index : int, optional
            cut-off end index. The default is None.
        history_size : int
            length of time series history to be used for making predictions.
        target_size : int
            length of the to-be-predicted time series.
        step : int, optional
            number of steps taken into account; ~ 1/sampling frequency. The default is 1.

        Returns
        -------
        None.

        """

        data = []
        labels = []
    
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(self._dataset) - target_size
    
        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.append(self._dataset[indices])

            labels.append(target[i:i+target_size])

        return np.array(data), np.array(labels)
    

    def create_time_steps(length: int):
        """ helper function for multi_stop_plot method
        """
        return list(range(-length, 0))
    
    
    def multi_step_plot(self, history: np.ndarray, true_future: np.ndarray
                        , prediction: np.ndarray) -> None:
        """ plot predicted and true values
        

        Parameters
        ----------
        history : np.ndarray
            DESCRIPTION.
        true_future : np.ndarray
            DESCRIPTION.
        prediction : np.ndarray
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        
        plt.figure(figsize=(12, 6))
        num_in = Kind_of_Blue.create_time_steps(len(history))
        num_out = len(true_future)
    
        size = 2
        plt.scatter(num_in, np.array(history[:]), label='History', s=size
                    , c='k')
        plt.scatter(np.arange(num_out), np.array(true_future)
                    , label='True Future', s=size, marker='x', c='r')
        if prediction.any():
            plt.scatter(np.arange(num_out), np.array(prediction)
                        , label='Predicted Future', s=size, marker='+'
                        , c='b')
            
        plt.legend(loc='upper left')
        plt.show()
        
        return None
    
    
    def generate_train_and_val_data(self, future_target_size: int
                                    , past_history_size: int
                                    , batch_size: int = None
                                    , buffer_size: int = 100000
                                    , target_column: int=None) -> None:
        """ sets self._train_data and self._val_data in tensorflow model input format
        

        Parameters
        ----------
        future_target_size : int
            DESCRIPTION.
        past_history_size : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
   
        # set random seed for the shuffle results to be reproducable
        tf.random.set_seed(22)
        
        # set target data
        if not target_column:
            self._target = self._dataset[:] 
        else:
            print('ToDo: implement this step')
        
        # get sliced training and validation set
        x_train, y_train = self.slice_time_series_data(target=self._target
                                                       , start_index=0
                                                       , end_index=self._train_split
                                                       , history_size=past_history_size
                                                       , target_size=future_target_size)
        
        x_val, y_val = self.slice_time_series_data(target=self._target
                                                   , start_index=self._train_split
                                                   , end_index=None
                                                   , history_size=past_history_size
                                                   , target_size=future_target_size)

        print('training set shape: x:{}, y:{}'.format(str(x_train.shape)
                                                      , str(y_train.shape)))
        print('validation set shape: x:{}, y:{}'.format(str(x_val.shape)
                                                        , str(y_val.shape)))
        
        self._num_samples = x_train.shape[0]
        
        if not batch_size:
            batch_size = future_target_size
        
        # format data such that its acceptable input to the tensor flow model 
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self._train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
        
        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        self._val_data = val_data.batch(batch_size).repeat()
        
        
        return None
    
    
    def compile_model(self, model_type: str
                     , units: int=16, num_layers: int=2
                     , output_shape: int=None
                     , use_dropout: bool=True) -> None:
        """
        

        Parameters
        ----------
        units_1 : int, optional
            DESCRIPTION. The default is 4.
        units_2 : int, optional
            DESCRIPTION. The default is 8.
        units_output_layer : int
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        
        if model_type=='LSTM':
            # initialize LSTM as sequential model
            LSTM_model = tf.keras.models.Sequential()
            
            if not output_shape:
                output_shape = tuple(self._train_data._flat_shapes[1])[1]
            
            # setup model configuration
            input_shape = tuple(self._train_data._flat_shapes[0])[1:]
            LSTM_model.add(tf.keras.layers.LSTM(units=units, return_sequences=True
                                                , input_shape=input_shape))
    
            
            # add layers
            print('debugLSTM: should an LSTM layer or a Dense layer be added?')
            while (num_layers - 2) > 0:
                # LSTM_model.add(tf.keras.layers.Dense(units=units
                #                     , activation='relu')
                               # )
                LSTM_model.add(tf.keras.layers.LSTM(units=units
                                                    , return_sequences=True
                                                    , activation='relu'))
                if use_dropout:
                    LSTM_model.add(tf.keras.layers.Dropout(0.2))
                    
                num_layers = num_layers - 1
            
            # penultimate layer
            # LSTM_model.add(tf.keras.layers.Dense(units=units, activation='relu'))
            LSTM_model.add(tf.keras.layers.LSTM(units=units, activation='relu'))
            if use_dropout:
                LSTM_model.add(tf.keras.layers.Dropout(0.2))
            
            # output layer
            LSTM_model.add(tf.keras.layers.Dense(output_shape))  
            
            # compile model
            LSTM_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            
            # add model to models dictionary
            self._models['LSTM'] = LSTM_model
            
            
        if model_type=='RNN':
            """
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.SimpleRNN(128, input_shape=input_shape, activation = 'relu'))
            model.add(tf.keras.layers.Dropout(0.1))
            model.add(tf.keras.layers.Dense(64, activation = 'relu'))
            model.add(tf.keras.layers.Dropout(0.1))
            model.add(tf.keras.layers.Dense(16, activation = 'relu'))
            model.add(tf.keras.layers.Dropout(0.1))
            model.add(tf.keras.layers.Dense(1, activation='linear'))
            
            model.compile(loss = 'mean_squared_error',
                          optimizer = 'adam',
                          metrics = ['mse'])
            """

            # initialize RNN as sequential model
            RNN_model = tf.keras.models.Sequential()
            
            if not output_shape:
                output_shape = tuple(self._train_data._flat_shapes[1])[1]
            
            # setup model configuration
            input_shape = tuple(self._train_data._flat_shapes[0])[1:]
            RNN_model.add(tf.keras.layers.SimpleRNN(units=units
                                                    , input_shape=input_shape))
            
            # add layers
            while (num_layers - 2) > 0:
                RNN_model.add(tf.keras.layers.Dense(units=units
                                                        , activation='relu'))
                if use_dropout:
                    RNN_model.add(tf.keras.layers.Dropout(0.2))
                    
                num_layers = num_layers - 1
            
            # penultimate layer
            RNN_model.add(tf.keras.layers.Dense(units=units, activation='relu'))
            if use_dropout:
                RNN_model.add(tf.keras.layers.Dropout(0.2))
            
            # output layer
            RNN_model.add(tf.keras.layers.Dense(output_shape))  
            
            # compile model
            RNN_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            
            # add model to models dictionary
            self._models['RNN'] = RNN_model
            
        return None
    
    
    def fit_model(self, epochs: int, steps_per_epoch: int
                  , validation_steps: int, model_type: str) -> None:
        """
        

        Returns
        -------
        None
            DESCRIPTION.

        """
    
        # steps_per_epoch : number of training batches = uniqueTrainingData / batchSize
            # https://stackoverflow.com/questions/45943675/meaning-of-validation-steps-in-keras-sequential-fit-generator-parameter-list/45944225
        
        time_callback = TimeHistory()
        
        if model_type=='LSTM':
            model = self._models['LSTM']
            
        elif model_type=='RNN':
            model = self._models['RNN']
            
        history = model.fit(self._train_data, epochs=epochs
                                  , steps_per_epoch=steps_per_epoch
                                  , validation_data=self._val_data
                                  , validation_steps=validation_steps
                                  , callbacks=[time_callback])
        if model_type=='LSTM':
            self._histories['LSTM'] = history
            self._time_callbacks['LSTM'] = time_callback
            
        elif model_type=='RNN':
            self._histories['RNN'] = history
            self._time_callbacks['RNN'] = time_callback
        
        return None
    
    
    def plot_history(self, model_type: str) -> None:
        """
        

        Returns
        -------
        Nonr
            DESCRIPTION.

        """
        if model_type=='LSTM':
            history = self._histories['LSTM']
        if model_type=='RNN':
            history = self._histories['RNN']
        
        plt.plot(history.history['mse'], label='training loss (mse)')
        plt.plot(history.history['val_mse'], label='validation loss (mse)')
        plt.legend()
        plt.show()
        
        return None
    
