\name{KFKSDS-package}
\docType{package}
\alias{KFKSDS-package}

\title{Kalman Filter, Smoother and Disturbance Smoother}

\description{
This package provides an implementation of the Kalman filter, 
smoother and disturbance smoother for state space models. 
An interface to run implementations of the filter that are 
available in other packages is also provided.
}

\details{
There are several other packages in \R that perform Kalman filtering and 
smoothing. A review can be found in Tusell (2011).
Each one has its own strengths that make them best suited to a particular context.
The package \pkg{KFKSDS} was developed as a tool to conduct the work described 
in López-de-Lacalle (2013a, 2013b).
Within this framework, the package is useful as a development tool as well 
as for debugging and testing purposes.

The implementation is naive in that it is a direct transcription of the equations 
of the filter and the smoother that can be found in many textbooks.
A square root filter is not considered in order to deal with potential 
numerical problems that may arise in the original equations.

The package includes a wrapper function that links to functions 
from the following packages:
\pkg{dlm}, \pkg{dse}, \pkg{FKF}, \pkg{KFAS} 
%\pkg{sspir}
and \pkg{stats}.
The original interface provided by each package is recommended as they 
sometimes provide further capabilities or options.
Nevertheless, the wrapper function provided in this package is useful 
for debugging and testing since it allows running different 
implementations of the filter through a unified interface.

A useful utility of the package is that it computes the analytical
derivatives of some of the elements of the filter and the smoother with 
respect to the parameters of the model.
In particular, the necessary elements to compute the analytical derivatives of 
the time domain log-likelihood function are returned. This is especially useful 
when it comes to maximizing the likelihood function.

%The matrices of the state space representation are considered time invariant.
%pure variance models
In some models, the Kalman filter and the smoother are expected to converge to 
a steady state. Some optional parameters can be defined in order to assess at each 
iteration of the filter whether a steady state has been reached.
When the steady state is reached, the values from the last iteration are used 
in the remaining iterations of the filter and smoother. Thus, the number of 
matrix operations can be reduced substantially.

Some parts of the code are implemented in the C language where the matrix operations
are handled through the GNU Scientific Library.

The package \pkg{stsm} is a useful complement to easily create an object 
containing the matrices of the state space representation for the structural 
models defined in that package.
}

\author{
Javier López-de-Lacalle \email{javlacalle@yahoo.es}
}

\section{References}{
Durbin, J. and Koopman, S. J. (2001).
\emph{Time Series Analysis by State Space Methods}. 
Oxford University Press.

Galassi, M., Davies, J., Theiler, J., Gough, B., Jungman, G., Booth, 
M. and Rossi, F. (2009).
\emph{GNU Scientific Library Reference Manual}. Network Theory Ltd.

Harvey, A. C. (1989).
\emph{Forecasting, Structural Time Series Models and the Kalman Filter}. 
Cambridge University Press.

López-de-Lacalle, J. (2013a).
\sQuote{101 Variations on a Maximum Likelihood Procedure for a Structural Time Series Model.}
Unpublished manuscript.

López-de-Lacalle, J. (2013b).
\sQuote{Why Does the Expectation-Maximization Algorithm Converge 
Slowly in Pure Variance Structural Time Series Models?}
Unpublished manuscript.

Tusell. F. (2011).
\sQuote{Kalman Filtering in \R.} 
\emph{Journal of Statistical Software}, \bold{39}(2). 
\url{http://www.jstatsoft.org/v39/i02/}.
}

\keyword{package, ts}
