# MNRAS: MN-16-2332-MJ referee report
## ABC LSS paper

Dear Editor,

This was a very constructive report and we have made revision to the paper accordingly. 
We think the comments have led to substantial improvement in quality of the paper.

ChangHooh Hahn & Mohammadjavad Vakili (on behalf of the authors):

We are very grateful to the constructructive comments made by the anonymous referee. 
They have substantially improved the paper. In what follows, the referee comments are 
indented and our responses are not indented.

Reviewer's Comments:

>The paper demonstrates how ABC can be used to constrain parameters of the HOD model.
The authors used a simulated "observed" catalog (for which the fiducial parameters are known) and show that ABC is able to provide model parameter constraints similar to those obtained from a pseudo-likelihood function.

>The paper is well written and the mathematical demonstrations are clear and didactic.

>I am also very keen to the ABC approach and firmly believe that it represents a step forward in the standard set of statistical approaches to cosmological problems.
In this context, I fully support efforts which aim to popularize the method within the astronomical community.

> However, as the authors themselves pointed out, the exercise of choosing a simulated "observed" catalog in order to show that the ABC approach leads to results which are consistent with the standard likelihood approach have already been extensively reported in the literature. Thus, I do not believe that the exercise by itself contains the level of novelty necessary to justify a publication in MNRAS.

The main reason that the referee cites is that our comparison between the standard approach and ABC using simulations is "extensively reported in the literature." In the context of cosmology, I don't think this is true. But I agree with Andrew that there's no reason to argue, especially when the referee only lists a few points to include. 

> I list below a few points which, if included in the discussion, will certainly contribute to the development of ABC in the astronomical context, and thus make the paper suitable for MNRAS:

>1) a deeper discussion on why the ABC approach should be preferred in this particular context (one or two phrases on this should be included in the abstract).
   in section 3.4 it is said:
   "Furthermore, the Gaussian-likelihood approach relies on constructing an accurate covariance
matrix estimate that captures the sample variance of the data. While we are able to
do this accurately within the scope of the HOD framework, for more general LSS parameter
inference situations, it is both labor and computationally expensive and dependent on the
accuracy of simulated mock catalogs, which are known to be unreliable on small scales (see
Heitmann et al. 2008; Chuang et al. 2015 and references therein). Since ABC-PMC utilizes
a forward model to account for sample variance, it does not depend on a covariance matrix
estimate; hence it does not face these problems."

>   this might lead to confusion. Do the authors believe that ABC presents no advantage  in the specific case studied in this paper?

I agree and I think it would be helpful to further motivate using ABC over MCMC, specifically in the context of cosmology. In other words, explicitly state in the abstract that the LF cannot actually be Gaussian and that an incorrect form of the LF can lead to biased parameter inference. This should also be further stressed in the intro with citations to papers such as  http://adsabs.harvard.edu/abs/2016MNRAS.456L.132S .

Our response to the comment: We agree with the referee's comment and we think it is helpful to further 
motivate ABC over MCMC, specially in the context of cosmology. In other words, we explicitly state in the 
abstract that the likelihhod function cannot be Gaussian and that an incorrect form of the likelihood function can 
lead to biased parameter inference. Therefore, we made significant changes to Section 3.4 (Comparison to MCMC analysis) in order to better emphasize the advantage of ABC over MCMC. In particular, we made made substantial changes to the paragraph mentioned in the referee's comment to address this:

    """ Accurate estimation of the covariance matrix 
      in LSS, however, faces a number of challenges. It is both labor and computationally 
      expensive and dependent on the accuracy of simulated mock catalogs, known to be 
      unreliable on small scales~(see Heitmann et al. 2008; Chuang et al. and references therein). 
      In fact, as Sellentin & Heaven (2016) points out, using estimates of the covariance 
      matrix in the Gaussian psuedo-likelihood approach become further problematic. Even 
      when inferring parameters from a Gaussian-distributed data set, using covariance 
      matrix estimates rather than the {\em true} covariance matrix leads to a likelihood 
      function that is {\em no longer} Gaussian. ABC-PMC does not depend on a covariance 
      matrix estimate; hence, it does not face these problems.
    """    

>2) if the Gaussian hypothesis underlying the standard approach are so unrealistic as stated in the text, why results are still consistent with the ABC ones? how likely it is that we will face a real data scenario which might lead to significantly different results?


I think that this comment can be addressed in the discussion by emphasizing that the consistency we find in our comparison is *not* validation of the Gaussian LF assumption. Instead, it results from the fact that our comparison is apparently not the most sensitive to it. Smaller scales and larger scales (as MJ and Hogg mentioned) -- both of which are very likely in "real data scenarios" -- will likely be more discerning. 

I also think this comment is important to address because readers might very well conclude from our paper, as it is now, that the Gaussian LF assumption is correct without understanding the nuanced conclusion.

>3) is it possible to simulate an extreme data situation where the results from the standard analysis are not consistent with the ABC ones? if so,  how realistic it is?

As Hogg and MJ mentioned, examining galaxy clustering at large (BAO) scales would be one possible way. I'm confused by the wording of this comment ... but that would be a *very* realistic scenario. This comparison would require a different approach than the scope of our paper. 


>4) what caveats one would face in trying to  apply this kind of analysis in real data? How can they affect the final results and how can they be circumvented?


Again, I'm a bit confused by the wording of this comment. Is the referee asking about the practical challenges of implementing ABC for parameter inference on real data? If so, I think the challenges would mainly be computational (i.e. the forward model). So the computation section should elaborate more on the caveats of scaling up our HOD forward model to a full blown cosmological one. 

Besides the computation, MJ and I came to the conclusion that most of the difficulty would be in the accuracy of the forward model. For instance, how well the forward model reproduces systematic effects. In the case of fiber collisions (my personal favorite), on BOSS it would be straightforward. On DESI, forward modeling the fiber positioners may be not so much. I don't think these issues are relevant for implementing ABC, however, because they also plague the standard approach. The systematic effect either has to be "corrected out" of the observation or the theoretical predictions have to model them -- I literally had to fiber collide *thousands* of mocks. 
