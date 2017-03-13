# MNRAS: MN-16-2332-MJ referee report
## ABC LSS paper

Dear Editor,


We thank the referee for the thorough review of our work and for the constructive report that substantially improved the quality of the paper.

Please find below responses to the referee's comments for the submission. 

Thank you, 
ChangHooh Hahn & Mohammadjavad Vakili (on behalf of the authors)

=================================
Reviewer's Comments:

>The paper demonstrates how ABC can be used to constrain parameters of the HOD model.
The authors used a simulated "observed" catalog (for which the fiducial parameters are known) and show that ABC is able to provide model parameter constraints similar to those obtained from a pseudo-likelihood function.

>The paper is well written and the mathematical demonstrations are clear and didactic.

>I am also very keen to the ABC approach and firmly believe that it represents a step forward in the standard set of statistical approaches to cosmological problems.
In this context, I fully support efforts which aim to popularize the method within the astronomical community.

> However, as the authors themselves pointed out, the exercise of choosing a simulated "observed" catalog in order to show that the ABC approach leads to results which are consistent with the standard likelihood approach have already been extensively reported in the literature. Thus, I do not believe that the exercise by itself contains the level of novelty necessary to justify a publication in MNRAS.

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


We agree with the referee's comment and have further motivated ABC in LSS cosmology throughout the text. We explicitly state in the abstract that the likelihood function form cannot, in detail, be Gaussian and that an incorrect form of the likelihood function can lead to biased parameter inference. 

We have also made significant changes to the discussion in Section 3.4 in order to better emphasize the advantage of ABC over MCMC. In particular, we have made substantial changes to the paragraph mentioned in the comment. We now emphasize that the ABC-PMC method accounts for sample variance in the generative forward model, whereas the pseudo-Gaussian likelihood method relies on the estimated covariance matrix. We also cite Sellentin & Heaven (2016), which finds that estimated covariance matrix leads to a likelihood function that is no longer Gaussian.

Furthermore, in the following paragraph, we emphasize the advantages of ABC-PMC method in dealing with systematics in parameter inference. Systematics can impact the likelihood function, but in ABC-PMC they can be simulated and marginalized out by the generative forward model. 

>2) if the Gaussian hypothesis underlying the standard approach are so unrealistic as stated in the text, why results are still consistent with the ABC ones? how likely it is that we will face a real data scenario which might lead to significantly different results?

We have added more detailed discussion of the discrepancy between the parameter constraints of ABC-PMC and the Gaussian pseudo-likelihood analysis in Section 3.4. We emphasize that the Gaussian pseudo-likelihood assumption for the two-point correlation function is incorrect because the correlation function must satisfy non-trivial positive-definiteness requirements. We expect this to lead to more discrepancy on very large scales (e.g. BAO scale and beyond) where there are fewer modes. However, investigation of the large-scale clustering is beyond the scope of this paper. 

We also emphasize that the GMF cannot be Gaussian-distributed as it is the abundance of galaxy groups. We argue that this incorrect assumption about the GMF likelihood may explain why the constraints on the parameter alpha are less biased for the ABC-PMC analysis than the Gaussian pseudo-likelihood analysis in Figure 8. 

>3) is it possible to simulate an extreme data situation where the results from the standard analysis are not consistent with the ABC ones? if so,  how realistic it is?

We have made substantial additions to the text in Section 3.4 to address how many data situations in LSS analysis can cause discrepancies between the standard analysis and the ABC-PMC method. For instance, we mention how luminosity or redshift incompleteness, selection functions, or observational systematics can imapct the likelihod function and lead to significant biases in a likelihood analysis. Then we emphasize how in the ABC-PMC method, these factors can be accounted for in the generative forward model of the simulated data without having to write down a likelihood function. 

>4) what caveats one would face in trying to  apply this kind of analysis in real data? How can they affect the final results and how can they be circumvented?


Again, I'm a bit confused by the wording of this comment. Is the referee asking about the practical challenges of implementing ABC for parameter inference on real data? If so, I think the challenges would mainly be computational (i.e. the forward model). So the computation section should elaborate more on the caveats of scaling up our HOD forward model to a full blown cosmological one. 

Besides the computation, MJ and I came to the conclusion that most of the difficulty would be in the accuracy of the forward model. For instance, how well the forward model reproduces systematic effects. In the case of fiber collisions (my personal favorite), on BOSS it would be straightforward. On DESI, forward modeling the fiber positioners may be not so much. I don't think these issues are relevant for implementing ABC, however, because they also plague the standard approach. The systematic effect either has to be "corrected out" of the observation or the theoretical predictions have to model them -- I literally had to fiber collide *thousands* of mocks. 
