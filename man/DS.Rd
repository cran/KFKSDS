\name{DS}
\alias{DS}
\alias{DS.deriv}

\title{Disturbance Smoother for State Space Models}

\description{
These functions run the disturbance smoother upon the output from the 
Kalman filter and smoother.
}

\usage{
DS(y, ss, kf, ks)
DS.deriv(ss, ksd)
}

\arguments{
\item{y}{a numeric time series or vector.}
\item{ss}{a list containing the matrices of the state space model.}
\item{kf}{a list containing the output returned by the function \code{KF}.}
\item{ks}{a list containing the output returned by the function \code{KS}.}
\item{ksd}{a list containing the output returned by the function \code{KS.deriv}.}
}

\details{
See the details section and the section \sQuote{state space representation}
in \code{\link[KFKSDS]{KF}}.
}

\value{
\code{DS} returns a list containing the following elements:
\item{epshat}{smoothed estimate of the disturbance term in the observation equation.}
\item{vareps}{error variance of \code{epshat}.}
\item{etahat}{smoothed estimate of the disturbance term(s) in the state equation.}
\item{vareta}{error variance of \code{etahat}.}

\code{DS.deriv} returns a list containing the derivatives of the elements above
named respectively \code{depshat}, \code{dvareps}, \code{detahat} and \code{dvareta}.
The derivatives are summed over all the observations.
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
\code{\link[stsm]{char2numeric}} in package \pkg{stsm}.
}

\examples{
# local level plus seasonal model with arbitrary parameter values
# for the 'JohnsonJohnson' time series
m <- stsm::stsm.model(model = "llm+seas", y = JohnsonJohnson, 
  pars = c("var1" = 2, "var2" = 15, "var3" = 30))
ss <- stsm::char2numeric(m)

kf <- KF(m@y, ss)
ks <- KS(m@y, ss, kf)
ds <- DS(m@y, ss, kf, ks)
acf(ds$epshat, main = "ACF of smoothed disturbance")

kfd <- KF.deriv(m@y, ss)
ksd <- KS.deriv(m@y, ss, kfd)
dsd <- DS.deriv(ss, ksd)

# compare analytical and numerical derivatives
fcn <- function(x, model, type, i = 1)
{
  m <- stsm::set.pars(model, x)
  ss <- stsm::char2numeric(m)
  kf <- KF(m@y, ss)
  ks <- KS(m@y, ss, kf)
  ds <- DS(m@y, ss, kf, ks)
  
  switch(type,
    "epshat" = sum(ds$epshat),
    "vareps" = sum(ds$vareps))
}

d <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "epshat")
all.equal(d, dsd$depshat)

d <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "vareps")
all.equal(d, dsd$dvareps)
}

\keyword{ts, model}
