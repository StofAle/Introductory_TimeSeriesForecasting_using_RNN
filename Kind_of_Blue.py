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
        self._TRAIN_SPLIT = None
        
        # training and validation data
        self._train_data = None
        self._val_data = None
        
        # target
        self._target = None  # is assigned in generate_train_and_val_data()
        
        # model
        self._model = None
        self._MODEL_TYPE = None
        
        # fitted model history
        self._history = None
        
        
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
                
                self._TRAIN_SPLIT = int(len(self._df) * Kind_of_Blue.TRAIN_SPLIT_RATIO)
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
        data_mean = self._dataset[:self._TRAIN_SPLIT].mean(axis=0)  
        data_std = self._dataset[:self._TRAIN_SPLIT].std(axis=0)
        
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
    
        plt.plot(num_in, np.array(history[:]), label='History')
        plt.plot(np.arange(num_out), np.array(true_future), 'bo',
               label='True Future')
        if prediction.any():
            plt.plot(np.arange(num_out), np.array(prediction), 'ro',
                     label='Predicted Future')
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
                                                       , end_index=self._TRAIN_SPLIT
                                                       , history_size=past_history_size
                                                       , target_size=future_target_size)
        
        x_val, y_val = self.slice_time_series_data(target=self._target
                                                   , start_index=self._TRAIN_SPLIT
                                                   , end_index=None
                                                   , history_size=past_history_size
                                                   , target_size=future_target_size)

        print('training set shape: x:{}, y:{}'.format(str(x_train.shape)
                                                      , str(y_train.shape)))
        print('validation set shape: x:{}, y:{}'.format(str(x_val.shape)
                                                        , str(y_val.shape)))
        
        if not batch_size:
            batch_size = future_target_size
        
        # format data such that its acceptable input to the tensor flow model 
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self._train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
        
        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        self._val_data = val_data.batch(batch_size).repeat()
        
        
        return None
    
    
    def compile_LSTM_model(self, units_1: int=4, units_2: int=8) -> None:
        """
        

        Parameters
        ----------
        units_1 : int
            DESCRIPTION.
        units_2 : int
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        
        # initialize LSTM as sequential model
        LSTM_model = tf.keras.models.Sequential()
        
        # setup model configuration
        input_shape = tuple(self._train_data._flat_shapes[0])[1:]
        LSTM_model.add(tf.keras.layers.LSTM(units_1, return_sequences=True
                                            , input_shape=input_shape))
        LSTM_model.add(tf.keras.layers.LSTM(units_2, activation='relu'))
        LSTM_model.add(tf.keras.layers.Dense(1))  # output layer
        
        # compile model
        LSTM_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        
        self._model = LSTM_model
        self._MODEL_TYPE = 'LSTM'
        
        return None
    
    
    def fit_model(self, epochs: int, steps_per_epoch: int
                  , validation_steps: int) -> None:
        """
        

        Returns
        -------
        None
            DESCRIPTION.

        """
    
        # steps_per_epoch : number of training batches = uniqueTrainingData / batchSize
            # https://stackoverflow.com/questions/45943675/meaning-of-validation-steps-in-keras-sequential-fit-generator-parameter-list/45944225
        history = self._model.fit(self._train_data, epochs=epochs
                                  , steps_per_epoch=steps_per_epoch
                                  , validation_data=self._val_data
                                  , validation_steps=validation_steps)
        
        self._history = history
        
        return None
    
    
    def plot_history(self) -> None:
        """
        

        Returns
        -------
        Nonr
            DESCRIPTION.

        """
        
        plt.plot(self._history.history['mse'], label='training loss (mse)')
        plt.plot(self._history.history['val_mse'], label='validation loss (mse)')
        plt.legend()
        plt.show()
        
        return None
    
