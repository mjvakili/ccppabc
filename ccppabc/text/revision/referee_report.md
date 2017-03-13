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


We agree with the referee's comment and have further motivated ABC in LSS cosmology throughout the text. In the abstract, we now explicitly state that the likelihood function form cannot, in detail, be Gaussian and that an incorrect form of the likelihood function can lead to biased parameter inference. 

We have also made significant changes to the discussion in Section 3.4 in order to better emphasize the advantage of ABC over MCMC. In particular, we have made substantial changes to the paragraph mentioned in the comment. We now emphasize that the ABC-PMC method accounts for sample variance in the generative forward model, whereas the pseudo-Gaussian likelihood method relies on the estimated covariance matrix. We also cite Sellentin & Heaven (2016), which finds that estimated covariance matrix leads to a likelihood function that is no longer Gaussian.

In the following paragraphs, we emphasize other advantages of the ABC-PMC method, such as dealing with systematics in parameter inference. Systematics can impact the likelihood function, but in ABC-PMC they can be simulated and marginalized out by the generative forward model. 

We now also reiterate in Section 3.4. how the Gaussian pseudo-likelihood assumption is incorrect for both the two-point correlation function and the group multiplicity function. 

>2) if the Gaussian hypothesis underlying the standard approach are so unrealistic as stated in the text, why results are still consistent with the ABC ones? how likely it is that we will face a real data scenario which might lead to significantly different results?

We have included more detailed discussion of the parameter constraint comparison between ABC-PMC and the Gaussian pseudo-likelihood methods. We emphasize that the GMF cannot be Gaussian-distributed as it is the abundance of galaxy groups and argue that this incorrect assumption about the GMF likelihood may explain why the constraints on the parameter alpha are less biased for the ABC-PMC analysis than the Gaussian pseudo-likelihood analysis in Figure 8. We have also included discussion of how more realistic scenarios, such as systematic effects, present many factors that can generate more severe discrepancies between the two methods. 

>3) is it possible to simulate an extreme data situation where the results from the standard analysis are not consistent with the ABC ones? if so,  how realistic it is?

In order to address this comment, we have made substantial additions to Section 3.4 discussing how many data situations in LSS analysis can cause discrepancies between the standard analysis and the ABC-PMC method. For instance, we mention how luminosity or redshift incompleteness, selection functions, or observational systematics can imapct the likelihod function and lead to significant biases in a likelihood analysis. We underline how in ABC-PMC, these factors can be accounted for in the generative forward model of the simulated data without having to write down a likelihood function. 

>4) what caveats one would face in trying to  apply this kind of analysis in real data? How can they affect the final results and how can they be circumvented?

We have included a more detailed discussion near the end of Section 3.4 that discusses some of the challenges in applying ABC-PMC parameter inference to real data. In particular, we discuss the computational challenges in using generative forward models that include cosmological simulations and robust forward modeling of  observational systematics. 