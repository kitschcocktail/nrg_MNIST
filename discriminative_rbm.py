#!/usr/bin/env python
import time

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy

import theano
import theano.tensor as T
import os
import cPickle

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from logistic_sgd import load_data


# start-snippet-1
class DRBM(object):
    """Discriminative Restricted Boltzmann Machine (DRBM)  """
    def __init__(
        self,
        input=None,
        label=None,
        n_visible=784,
        n_label=10,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        U=None, 
        dbias=None,
        numpy_rng=None,
        theano_rng=None
    ):
        """
        DRBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units
        
        :param n_label: number of label units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        
        :param U: shared weight matrix between hidden units and label units
        
        :param dbias: shared hidden units bias for label layer. 
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_label  = n_label

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )
        
        if U is None:
            # U is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_label+n_hidden)) and
            # 4*sqrt(6./(n_label+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_U = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_label)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_label)),
                    size=(n_label, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            U = theano.shared(value=initial_U, name='U', borrow=True)
        
        if dbias is None:
            # create shared variable for label hidden units bias
            dbias = theano.shared(
                value=numpy.zeros(
                    n_label,
                    dtype=theano.config.floatX
                ),
                name='dbias',
                borrow=True
            )

        # initialize input layer for standalone DRBM 
        self.input = input
        if not input:
            self.input = T.matrix('input')
        self.label = label
        if not label:
            self.label = T.matrix('label')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.U = U
        self.dbias = dbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias, self.U, self.dbias]
        # end-snippet-1

    def free_energy(self, v_sample, label_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + T.dot(label_sample, self.U) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias) + T.dot(label_sample, self.dbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis, lab):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + T.dot(lab, self.U) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample, lab0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles and the labels
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample, lab0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units and labels

        Note that we return also the pre_sigmoid_activation of the
        layers. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        pre_sigmoid_label = T.dot(hid, self.U.T) + self.dbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation),
                pre_sigmoid_label, T.nnet.softmax(pre_sigmoid_label)]
                # TODO T.nnet.softmax(pre_sigmoid_label) or T.nnet.sigmoid(pre_sigmoid_label)

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean, pre_sigmoid_lab1, lab1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        """lab1_sample = self.theano_rng.binomial(size=lab1_mean.shape,
                                             n=1, p=lab1_mean,
                                             dtype=theano.config.floatX)"""
        lab1_sample = T.round(lab1_mean)
        
                
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_lab1, lab1_mean, lab1_sample]
                 # TODO How to sample lab1 through lab1_mean?

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_lab1, lab1_mean, lab1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample, lab1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_lab1, lab1_mean, lab1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample, lab0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample, lab0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_lab1, lab1_mean, lab1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_lab1, lab1_mean, lab1_sample]

    # start-snippet-2
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input, self.label)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nlabs,
                nlab_means,
                nlab_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, None, None, None, chain_start],
            n_steps=k
        )
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]
        label_end = nlab_samples[-1]

        cost = T.mean(self.free_energy(self.input, self.label)) - T.mean(
            self.free_energy(chain_end, label_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end, label_end])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )

        return cost, updates
        # end-snippet-4 
    
    
          
    def save_model(self, filename='DRBM.pkl',
                   save_dir='output_folder'):
        # Save the parameters of the model 

        print '... saving model'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_file = open(os.path.join(save_dir, filename), 'wb')
        cPickle.dump(self.params, save_file, protocol=cPickle.HIGHEST_PROTOCOL)
        save_file.close()

    
def test_drbm(learning_rate=0.1, training_epochs=20,
             dataset='mnist.pkl.gz', batch_size=20, 
             output_folder='rbm_plots', n_hidden=500):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    
    # modify the structure of the training labels
    seed = numpy.zeros((train_set_x.eval().shape[0], 10))
    for i in range(train_set_x.eval().shape[0]):
        seed[i][train_set_y.eval()[i]] = 1
    
    train_set_y = theano.shared(numpy.asarray(seed, dtype=theano.config.floatX))

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('label')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    
    # construct the RBM class
    drbm = DRBM(input=x, label=y, n_visible=28 * 28,
              n_hidden=n_hidden, n_label=10, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-1
    cost, updates = drbm.get_cost_updates(lr=learning_rate,
                                         persistent=None, k=5)

                                         
                                         
    #################################
    #     Training the RBM          #
    #################################
    
    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = time.clock()

    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
        
        # save the parameters of RBM 
        os.chdir('/Users/blanco/desktop/prueba_generate')
        drbm.save_model()
        
        
        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)
        
        os.chdir(output_folder)

        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=drbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))
    

if __name__ == '__main__':
    test_drbm()
