#!/usr/bin/env python

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy
import os
import time
import cPickle
import matplotlib.pyplot as plt
from pylab import cm

from theano.tensor.shared_randomstreams import RandomStreams



from logistic_sgd import load_data            
dataset = 'mnist.pkl.gz'
datasets = load_data(dataset)
test_set_x, test_set_y = datasets[2]

true_ys = numpy.array(test_set_y.eval()[:1000])
test_0 = numpy.array(test_set_x.get_value(borrow=True)[9])
test_0imp = ran_1(test_0, 8)

digit_weight = numpy.mean(test_set_x.get_value(borrow=True), axis=0)
inv_weight = 1 - digit_weight


def img_save(digit_vector, savename):
    plt.imshow(-digit_vector.reshape(28,28), cmap=cm.gray)
    plt.xticks([])
    plt.yticks([])
    os.chdir(os.getcwd() + "/test_results")
    plt.savefig(savename)
    os.chdir('../')



    
def stats(classn, FE_pn, true_y):
    def err(classes_a, classes_b):
        same = []
        for y_a, y_b in zip(classes_a, classes_b):
            same.append((y_a == y_b).astype(float))
        pc_same = 1 - numpy.mean(same)
        return pc_same
        
    print 'error', err(classn, true_y)
    print 'max', numpy.max(FE_pn, axis=0)
    print 'mean', numpy.mean(FE_pn, axis=0)
    
# find the free energy range for 0% imputation


class0 = [] ; FE_p0 = []
for i in range(1000):
    print i
    dclass, FE_p, recon_digit = classify_r(test_set_x.get_value(borrow=True)[i])
    class0.append(dclass)
    FE_p0.append(FE_p)
stats(class0, FE_p0, true_ys)


class2 = [] ; FE_p2 = [] ; recon2 = []
for i in range(1000):
    print i
    dclass, FE_p, recon_dgt = classify_r(test_set_x.get_value(borrow=True)[i],\
                                         reclass=True,imp_pc=2)
    class2.append(dclass)
    FE_p2.append(FE_p)
    recon2.append(recon_dgt)
stats(class2, FE_p2, true_ys)

class2rr = [] ; FE2rr = []
for recon_img in recon2:
    dclass, FE_p, recon_dgt = classify_r(recon_img, reclass=True)
    class2rr.append(dclass)
    FE2rr.append(FE_p)
stats(class2rr, FE2rr, true_ys)


class8 = [] ; FE_p8 = []
start_time = time.clock()
for i in range(1000):
    print i
    dclass, FE_p, recon_dgt = classify_r(test_set_x.get_value(borrow=True)[i],\
                                         reclass=True, imp_pc=8)
    class8.append(dclass)
    FE_p8.append(FE_p)
end_time = time.clock()
print 'Time: %.2fm' % ((end_time - start_time) / 60.)
stats(class8, FE_p8, true_ys)

# compare 0% and 2% free energy
FE0vs2 = []
for fe0, fe2 in zip(FE0[:,0], FE2[:,0]):
    FE0vs2.append((fe0 - fe2))
FE0vs2 = -numpy.mean(FE0vs2) # 101.89 avg increase








#-----------------------------------------------------------------------
# impute data at 4%
imp_4 = numpy.array([ran_1(digit_img, 4) for digit_img in test_xs])
# get details from classification task
class4 = []; FE4 = [];
for imp_digit in imp_4:
    dclass, FE, p, dgt_img = classify_r(imp_digit)
    class4.append(dclass) # discovered classes
    FE4.append([FE,p]) # free energies
FE4_mean = numpy.mean(FE4, axis=0)
FE4_max = numpy.max(FE4, axis=0)
FE4 = numpy.array(FE4)

# comparisons
# 4% imp vs. true y error
err4 = err(class4, true_ys) # 0.18 error
# compare 0% and 2% free energy
FE0vs4 = []
for fe0, fe4 in zip(FE0[:,0], FE4[:,0]):
    FE0vs4.append((fe0 - fe4))
FE0vs4 = -numpy.mean(FE0vs4) # 102.3 average increase

#reclassified
class4r = [] ; FE4r = [] ; imp_4reconstructed = []
for imp_digit in imp_4:
    dclass, FE, p, recon_digit = classify_r(imp_digit, threshold=-270)
    class4r.append(dclass)
    FE4r.append([FE, p])
    imp_4reconstructed.append(recon_digit)
err4r = err(class4r, true_ys) # .14 error
FE4r_mean = numpy.mean(FE4r, axis=0)
FE4r_max = numpy.max(FE4r, axis=0)

# reconstructed
class4rr = [] ; FE4rr = []
for recon_img in imp_4reconstructed:
    dclass, FE, p, recon_digit = classify_r(recon_img, threshold=-270)
    class4rr.append(dclass)
    FE4rr.append([FE, p])
err4rr = err(class4rr, true_ys) # .14 error
FE4rr_mean = numpy.mean(FE4rr, axis=0)
FE4rr_max = numpy.max(FE4rr, axis=0)

#reconstructed + mixed with original imputed dataset, but not reclassified
imp_4rm = numpy.array([(a + b) / 2 for a, b in zip(imp_4, imp_4reconstructed)])
class4rm = [] ; FE4rm = []
for imp_digit in imp_4rm:
    dclass, FE, p, recon_digit = classify_r(imp_digit)
    class4rm.append(dclass)
    FE4rm.append([FE, p])
err4rm = err(class4rm, true_ys) 
FE4rm_mean = numpy.mean(FE4rm, axis=0)
FE4rm_max = numpy.max(FE4rm, axis=0)

#reconstructed + mixed with original imputed dataset, + reclassified
class4rmr = [] ; FE4rmr = []
for imp_digit in imp_4rm:
    dclass, FE, p, recon_digit = classify_r(imp_digit, threshold=-270)
    class4rmr.append(dclass)
    FE4rmr.append([FE, p])
err4rmr = err(class4rmr, true_ys) # .14 error
FE4rmr_mean = numpy.mean(FE4rmr, axis=0)
FE4rmr_max = numpy.max(FE4rmr, axis=0)


#-----------------------------------------------------------------------

# impute 25% of test set
how_many = numpy.random.randint(0, test_xs.shape[0], 25)
imp_mix = list(test_xs)
for i in how_many:
    imp_mix[i] = ran_1(imp_mix[i], 2)

classmix = []
for digit in imp_mix:
    dclass = classify_r(digit, -270)
    classmix.append(dclass) # discovered classes
errm = err(classmix, true_ys) # .12 error



#-------------------------------------------------------------------------
# generate images with discovered classes
CDs = 100
gen_dgt2 = numpy.array([generate(dgts, c_l, CDs)\
                for dgts, c_l in zip(wts2, cl2)])
# get details from classification task with generated digit images
class2_2 = []; FE2_2 = [] ; wts2_2 = [] ; cl2_2 = []
for imp_digit in gen_dgt2:
    dclass, FE, wts, cl = classify_MNIST(imp_digit)
    class2_2.append(dclass)
    FE2_2.append(FE)
    wts2_2.append(wts)
    cl2_2.append(cl)


#------------------------------------------------------------------------
# comparisons
# 2% imp round 2 vs. true y error 
err2_2vs0 = err(class2_2, true_ys) # 0.25 error
FE0vs2_2 = []
for fe0, fe2_2 in zip(FE0, FE2_2):
    FE0vs2_2.append((fe0 - fe2_2))
FE0vs2_2 = - numpy.mean(FE0vs2_2) # 2.5 average increase

err2_2vs2 = err(class2_2, class2) # .12 error
FE2vs2_2 = []
for fe2, fe2_2 in zip(FE2, FE2_2):
    FE2vs2_2.append((fe2 - fe2_2))
FE2vs2_2 = - numpy.mean(FE2vs2_2) # -96.9 average increase


#------------------------------------------------------------------------
# mix reconstructed images and 2% imputed images and get details
class2_2mix = []; FE2_2mix = [] ; wts2_d2mix = [] ; cl2_2mix = []
mix = numpy.array([a+b for a, b in zip(imp_2, numpy.round(gen_dgt2, 3))])
for imp_digit in mix:
    dclass, FE, wts, cl = classify_MNIST(imp_digit)
    class2_2mix.append(dclass)
    FE2_2mix.append(FE)
    wts2_d2mix.append(wts)
    cl2_2mix.append(cl)
#-------------------------------------------------------------------------
 # comparisons
err2mix = err(class2_2mix, true_ys) # .21 error
err2mixvs1 = err(class2_2mix, class2) # .17 error
err2mixvs2 = err(class2_2mix, class2_2) # .22 error

# generate images with discovered classes, round 3
gen_dgt3= numpy.array([generate(dgts, c_l, CDs)\
          for dgts, c_l in zip(wts2_2, cl2_2)])

class2_3 = []; FE2_3 = []
for imp_digit in gen_dgt3:
    dclass, FE = classify_MNIST(imp_digit)[:2]
    class2_3.append(dclass)
    FE2_3.append(FE)
class2_3 = numpy.array(class2_3)

err2_3vs0 = err(class2_3, true_ys) # .3 error
FE0vs2_3 = numpy.array([(fe0 - fe2_2) for fe0, fe2_2 in zip(FE0, FE2_2)])
FE0vs2_3 = - numpy.mean(FE0vs2_3) # -13 average increase

err2_3vs2 = err(class2_3, class2) # .2 error
err2_3vs2_2 = err(class2_3, class2_2) # .09 error

gut2_FE_rnd1vs3 = []
for round1, round3 in zip(imp_2FEs, imp_2FEs_rnd3):
    gut2_FE_rnd1vs3.append((round1 > round3).astype(float))
decreased_1vs3 = numpy.mean(gut2_FE_rnd1vs3)

gut2_FE_rnd2vs3 = []
for round2, round3 in zip(imp_2FEs_rnd2, imp_2FEs_rnd3):
    gut2_FE_rnd2vs3.append((round2 > round3).astype(float))
decreased_2vs3 = numpy.mean(gut2_FE_rnd2vs3)


# plot results        
fig, axes = plt.subplots(1, 10, figsize=(10, 1),
                         subplot_kw={'xticks': [], 'yticks': []})
    
fig.subplots_adjust(hspace=0.0, wspace=0.0)    

index = 0
for ax, output in zip(axes.flat, imp_2[:10]):
    ax.imshow(-output.reshape(28,28), cmap=cm.gray)
    #title = round(MSE(visprob[0], output), 4)
    #subtitle = round(Es[index], 2)
    #ax.set_title(str(title) + '\n' + str(subtitle))
    index += 1
#plt.plot(Es)
plt.show()

#fig.show()
os.chdir('test_results')
fig.savefig('gen', bbox_inches='tight')
os.chdir('..')



print 1-numpy.mean(class_pc)
#print numpy.amax(FE_pc)
print numpy.mean(FE_pc)


#print numpy.amax(FE_pc_rr)
print numpy.mean(FE_pc_rr)
print 1-numpy.mean(class_pc_rr)

#print numpy.amax(FE_pc_r)
print numpy.mean(FE_pc_r)
print 1-numpy.mean(class_pc_r)

