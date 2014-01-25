\name{KFKSDS}
\alias{KSDS}
\alias{KSDS.deriv}
\alias{KFKSDS.deriv.C}
\alias{KFKSDS.deriv.steady.C}

\title{Kalman Filter, Smoother and Disturbance Smoother}

\description{
Kalman filter, smoother and disturbance smoother.
}

\usage{
KSDS(y, ss, kf)
KSDS.deriv(y, ss, kf)
KFKSDS.deriv.C(y, ss)
KFKSDS.deriv.steady.C(y, ss, convergence = c(0.001, 10, 1.2))
}

\arguments{
\item{y}{a numeric time series or vector.}
\item{ss}{a list containing the matrices of the state space model.}
\item{kf}{a list containing the output returned by the function \code{KF}.}
\item{convergence}{a numeric vector of length three to control and determine 
the convergence of the filter and smoother. See details below.}
}

\details{
See the details section and the section \sQuote{state space representation}
in \code{\link[KFKSDS]{KF}}.

The iteration at which the the Kalman smoother converges is determined as the 
iteration where the Kalman filter converged multiplied by the factor 
\code{convergence[3]}. It should be equal or greater than unity.
It can be omitted by setting it equal to one.

These interfaces are fully implemented in \R.

In these functions, the Kalman smoother and the disturbance smoother are run in a 
single loop instead of running two sepate loops over the same observations.

These functions return a relatively extended output than can be used, for example, 
to implement an expectation-maximization algorithm.
}

\value{
The function \code{KSDS} returns a list containing the following elements:
\item{ahat}{smoothed state disturbance.}
\item{varhat}{covariance matrix of \code{ahat}.}
\item{r}{weighted sum of innovations used to obtain \code{ahat}.}
\item{N}{intermediate matrix used to obtain \code{varahat}}
\item{epshat}{smoothed estimate of the disturbance term in the observation equation.}
\item{vareps}{error variance of \code{epshat}.}
\item{etahat}{smoothed estimate of the disturbance term(s) in the state equation.}
\item{vareta}{error variance of \code{etahat}.}

The function \code{KSDS-deriv} returns also \code{dahat}, \code{dvarahat}, 
\code{dr}, \code{dN} which are the derivatives referred to the elements defined above.

The functions \code{KFKSDS.deriv.C} and \code{KFKSDS.deriv.steady.C}
return a list containing the elements already defined above:
\code{epshat}, \code{vareps}, \code{etahat}, \code{vareta}, 
\code{r}, \code{N}, \code{dr}, \code{dN}, \code{dahat} and \code{dvareps}.
}

\references{
Durbin, J. and Koopman, S. J. (2001).
\emph{Time Series Analysis by State Space Methods}. 
Oxford University Press.

Harvey, A. C. (1989).
\emph{Forecasting, Structural Time Series Models and the Kalman Filter}. 
Cambridge University Press.
}

\seealso{
\code{\link[KFKSDS]{KF}}, \code{\link[KFKSDS]{KS}};
\code{\link[stsm.class]{char2numeric}} in package \pkg{stsm.class}.
}

\examples{
# See tests comparing the analytical derivatives returned by
# these functions and the numerical derivatives evaluated with
# function 'numDeriv::grad' in the folder 'KFKSDS/inst/tests'
# of the source package

# local level plus seasonal model with arbitrary parameter values
# for the 'JohnsonJohnson' time series
m <- stsm.class::stsm.model(model = "llm+seas", y = JohnsonJohnson, 
  pars = c("var1" = 2, "var2" = 15, "var3" = 30))
ss <- stsm.class::char2numeric(m)
kfd <- KF.deriv(m@y, ss)
ksds <- KSDS(m@y, ss, kfd)
da <- KSDS.deriv(m@y, ss, kfd)
db <- KFKSDS.deriv.C(m@y, ss)
# the ouput is the same but is returned with a different arrangement
dim(da$dahat)
dim(db$dahat)
for (i in seq_along(m@y))
  stopifnot(all.equal(da$dahat[i,,], db$dahat[,,i]))
dim(da$dN)
dim(db$dN)
for (i in seq_along(m@y))
  for (j in seq_len(3))
    stopifnot(all.equal(diag(da$dN[i,,,j]), db$dN[,j,i], check.attributes = FALSE))
}

\keyword{ts, model}
