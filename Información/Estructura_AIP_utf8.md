M. Berardi et al.
Computer Methods in Applied Mechanics and Engineering 435 (2025) 117628
The set??= {(??, ??)}? is the set of training points, and? is the number of training points. We highlight here that, since we
? ? ?=1
are going to compare the PINN solution to synthetic data, we select collocation points coincident with training points. We recall
that training points are used to teach the network to fit the known solution in the data-driven regions of the problem space, whereas
collocation points are used to ensure that the solution, provided by the neural network, respects the physical law modelled by the
differential equation considered (for further details we refer to [47]). Therefore, in general training points and collocation points
could be different collocation points are a subset of training points but, here, they will be given by the same set of points.
The chosen norm???(it may be different for each term in the loss function) depends on the functional space and the specific
problem. Selecting a correct norm (to avoid overfitting) for the loss function evaluation is an important problem in PINN, and
recently, in [48], the authors have proposed spectral techniques based on Fourier residual method to overcome computational and
accuracy issues. The first term in the right-hand side of Eq. (6)is referred to as data fitting loss and could possibly handle both initial
and boundary conditions, while the second term is referred to as residual loss, which is responsible for making the NN informed
by the physics of the problem. The derivatives inside ? in space, time and in the parameter space are usually performed using
autodiff(Automatic Differentiation algorithm, see [49,50]). Using the NN to approximate?in the loss function Eq. (6)allows us
to solve the PDE by minimising the loss function with respect to the parameters? of the NN. If??? ? ? (?), then the minimisation
problem can be written as
| ?å= ar g m | in?(? ? ? (?);? | ).  |     |     |     | (7) |
| ---------- | --------------- | --- | --- | --- | --- | --- |
| ?          |                 | 0   |     |     |     |     |
For a more detailed discussion on the PINN structure and the loss function, we refer to [51], [7] and to the review in [52].
3.1. Inverse PINN
Inverse PINNs are a type of Neural Network specifically designed to determine constitutive parameters or problem-related
functions that appear in the PDE one must solve. However, due to the limited amount of data relative to exact solutions, or of
available measurements of the physical problem described by the PDE underlying the PINN, the inverse problem(7)could likely
be ill-posed, and thus particular care has to be put into the training strategy during the optimisation process (see, e.g., [53]). In
particular, different contributions in the loss functions Eq. (6)could conflict with each other, providing an unbalanced gradient back-
propagation during the training, which would result in a troublesome convergence process [54]. Thus, several strategies have been
recently developed to cope with these issues: among the others, one could resort toGradNorm[55] to dynamically tune gradient
magnitudes to balance learning tasks; to PCGrad [56] to project each gradient on the tangent plane to all the other conflicting
gradients to mitigate such destructive interference; to Multi-Objective Optimisation [57]; to Self-Adaptive PINNs [58], where each
training point is weighed individually, so to penalise more points in difficult regions of the domain.
Using the notation introduced in the previous section, the inverse PINN minimisation now takes into account also the physical
parameters and can be written as
| [           |               |              |     | ]     |     |     |
| ----------- | ------------- | ------------ | --- | ----- | --- | --- |
| (?å, ? å) = | ar g m in ?(? | (?);? ) +??? | ??  | ? 2 , |     |     |
| 0           | ? ,? ? ?      | 0            | 0   | 0 ?   |     | (8) |
0
where?is a regularisation parameter. The second term in the right-hand side of the equation is the regularisation term, which is
used to prevent overfitting and to ensure that the physical parameters? are close to the some reference parameters??. If otherwise
|     |     |     |     |     | 0 0 |     |
| --- | --- | --- | --- | --- | --- | --- |
stated, in the following we will consider?= 0.
With reference to the mathematical models introduced in Section 2, we will consider the following physical parameters to be
estimated: the diffusion coefficient ? in the heat equation (1), the velocity ? and the dispersion coefficient ? in the advectionû
diffusionûreaction equation(3), and the transfer coefficient?in the mobileûimmobile model(4). The physical parameters will be
considered as trainable parameters in the NN, and the reference data will be added to the loss function Eq. (6).
3.2. Adaptive inverse PINN
To ensure the convergence of the inverse PINN, we redefine a weighted loss function as
?
| ?(?       | ) = ?( ?? | ???(?? , ?? ) ??? | 2+?? | ?(?(?? | , ?? 2)      |     |
| --------- | --------- | ----------------- | ---- | ------ | ------------ | --- |
| ? ? (?);? | 0         | ? ?               | ??   | ?? ?   | ? );? 0 )? , | (9) |
?=1
where?? are weight factors that depends on the training iteration?. The weights are updated at each iteration to ensure that the
?
different components of the loss function are balanced. The weights are updated using the following formula:
???
?
| ??=  | ,       | ?= 1,à, ? , |     |     |     | (10) |
| ---- | ------- | ----------- | --- | --- | --- | ---- |
| ? ?? | ???+??? |             |     |     |     |      |
| ?=1  | ? ?     |             |     |     |     |      |
???
| ??  | ?         |     |     |     |     |      |
| --- | --------- | --- | --- | --- | --- | ---- |
| ? = | ,         |     |     |     |     | (11) |
| ??  | ???+???   |     |     |     |     |      |
| ?=1 | ? ?       |     |     |     |     |      |
| ??  | if?? ?? ? |     |     |     |     |      |
| ? ? | ?         |     |     |     |     |      |
?
| ?? ? ?   | if?? = 0                         |     |     |     |             |      |
| -------- | -------------------------------- | --- | --- | --- | ----------- | ---- |
| ???=     | ?                                |     |     | ,   | ?= 1,à, ? , | (12) |
| ? ??(?)? | if(??, ??)is a collocation point |     |     |     |             |      |
| ?        | ? ?                              | ?   |     |     |             |      |
| ?0       | otherwise                        |     |     |     |             |      |
?
5

M. Berardi et al.
Computer Methods in Applied Mechanics and Engineering 435 (2025) 117628
|          | Fig. 3. | Qualitative behaviour of? | in(14)for?= 5000and? | = 1000. |      |
| -------- | ------- | ------------------------- | -------------------- | ------- | ---- |
|          |         |                           | ?                    | 0       |      |
| ??? = 1, |         |                           |                      |         | (13) |
?
, and are the weights for the boundary conditions, initial conditions, and collocation points, respectively. The
| where ? ? ? , ? ? ? | ? ? |     |     |     |     |
| ------------------- | --- | --- | --- | --- | --- |
is an increasing function of the epoch ?, such that = 0 ?. This allows the PINN to be trained
| function ? ? |     |     | ? ? and | ? ? ? 1 as ? ? |     |
| ------------ | --- | --- | ------- | -------------- | --- |
initially solely by the PDE residual. In the following, we will consider
( ( ))
t anh ????2??0 + 1
10
| ?(?) = | ?   | ,  ?= 1,à, ? , |     |     | (14) |
| ------ | --- | -------------- | --- | --- | ---- |
2
is the total number of epochs, and? is a threshold epoch before the weights are updated more significantly; seeFig. 3
| where? |     | 0   |     |     |     |
| ------ | --- | --- | --- | --- | --- |
for a typical graph of a function of this kind.
The gradients ? ? and ? ? are computed with the autodiff algorithm, and the latter (the gradients with respect to the
? ?0
physical parameters) are scaled by? ?(?)at each iteration. The scaling of the gradients is crucial for the convergence of the inverse
PINN, as it ensures that the physical parameters are updated only when data is included in the loss function. The parameters are
then updated with the Adam optimiser, with a sequence of learning rates that decrease at each iteration according to the epoch?.
Namely starting from a learning rate? at the first epoch, the learning rate is updated as:
0
? ? ?
| ? =? ? , |     |     |     |     | (15) |
| -------- | --- | --- | --- | --- | ---- |
| ? 0 100  |     |     |     |     |      |
where0.9< ? <0.99is a constant factor. An algorithmic description of the above process is given in Algorithm1.
Algorithm 1Training Algorithm with Adaptive Weights and Gradient Updates.
epoch = 0
1:
2: repeat
epoch = epoch + 1
3:
| 4: ifdo_par amet er_t r ainandepoch> ? |     | then |     |     |     |
| -------------------------------------- | --- | ---- | --- | --- | --- |
0
5: compute?(epoch)as in Eq.(14)
6: end if
7: update data weights as in Eq.(10)
8: compute gradients of loss function
| rescale gradients relative to? | by?(epoch) |     |     |     |     |
| ------------------------------ | ---------- | --- | --- | --- | --- |
| 9:                             | 0          |     |     |     |     |
apply gradients to all trainable parameters
10:
untilconvergenceorepoch>epochs
11:
4. Numerical results
In this section, we apply our PINN to different models arising from Eq. (3)and Eq. (4), under several assumptions and conditions.
All codes and data used in this work are published and freely available [39]. We report here a series of numerical experiments starting
from a random initial guess for the parameters, and we show the convergence of the PINN to the correct values. The robustness
of the approach with respect to the initial value of the parameters is shown in Appendix through two additional random initial
6
