\name{KF-interfaces}
\alias{KF-interfaces}
\alias{KalmanFilter}
\alias{make.KF.args}

\title{Kalman Filter}

\description{
This function provides a unified interface for implementations 
of the Kalman filter available in different packages.
}

\usage{
KalmanFilter(y, ss, 
  KF.version = c("KFKSDS", "StructTS", "KFAS", "FKF", "dlm", "dse"),
  KF.args = list(), check.args = TRUE, debug = FALSE)
make.KF.args(ss, KF.version, KF.args = list())
}

\arguments{
\item{y}{a numeric time series or vector.}
\item{ss}{a list containing the matrices of the state space model.}
\item{KF.version}{a character string indicating the implementation to be used.}
\item{KF.args}{a list containing optional arguments to be passed to the 
function that runs the Kalman filter.}
\item{check.args}{logical. If \code{TRUE}, it is checked that the elements 
of \code{KF.args} are valid for the selected interface \code{KF.version}.}
\item{debug}{logical. Currently ignored.}
}

\details{
For some purposes such as testing, debugging or development of extensions, 
this function provides a useful unified interface for different 
implementations of the Kalman filter.
%Each implementation of the filter has each own strengths and may be best 
%suited for particular contexts. 
%e.g. multivariate data, time varying matrices
In a production environment, the use of the original interfaces provided by each package 
is recommended since they sometimes provide further options or may incorporate 
further capabilities in the latest updates.

The elements in the argument \code{ss} must be named in accordance with 
the notation given in \code{\link[KFKSDS]{KF}}.
The function \code{\link[stsm]{char2numeric}} in package \pkg{stsm}
is a convenient way to create the argument \code{ss} for those models already 
defined in that package.

If \code{KF.args} is empty, default values are defined depending on the interface
selected in \code{KF.version}. The function \code{make.KF.args} set default values 
for those arguments that are explicitly defined in \code{KF.args}. It also 
checks that the arguments passed through \code{KF.args} are correct and a warning 
is given if any of them does not apply to the selected interface \code{KF.version}.

Argument \code{KF.version}: the option \code{StructTS} applies the Kalman filter as 
in the function \code{StructTS} of the \pkg{stats} package. The remaining possible 
values for this argument are the names of the package that contains 
the Kalman filter interface.

Notes: 
(1) The package \pkg{sppir} is no longer maintained on CRAN and is not currently
available here as an option. For old versions see 
\href{http://cran.r-project.org/src/contrib/Archive/sspir/}{sspir}.

(2) \code{KF.version="dse"} requires manually loading the package \pkg{dse}.
}

\value{
A list containing the output returned by each interface and the value
of the negative of the log-likelihood function in the element \code{mloglik}.
}

\references{
Dethlefsen, C., Lundbye-Christensen, S. and Christensen A. L.  (2012)
\R package version 0.2.10.
\emph{\pkg{sspir}: State Space Models in \R},
\url{http://CRAN.R-project.org/package=sspir}.

Durbin, J. and Koopman, S. J. (2001).
\emph{Time Series Analysis by State Space Methods}. 
Oxford University Press.

Gilbert, P. D. (2013) \R package version 2013.3-2.
\pkg{dse} \emph{Brief User's Guide: Dynamic Systems Estimation},
\url{http://CRAN.R-project.org/package=dse}.

Harvey, A. C. (1989).
\emph{Forecasting, Structural Time Series Models and the Kalman Filter}. 
Cambridge University Press.

Helske, J. (2012). \R package version 0.9.11.
\emph{\pkg{KFAS}: Kalman Filter and Smoother for Exponential family state space models.},
\url{http://CRAN.R-project.org/package=KFAS}.

Luethi, D., Erb, P. and Otziger, S. (2012) \R package version 0.1.2.
\emph{\pkg{FKF}: Fast Kalman Filter},
\url{http://CRAN.R-project.org/package=FKF}.

Petris, G. (2013) \R package version 1.1-3.
\pkg{dlm} \emph{An R Package for Dynamic Linear Models},
\url{http://CRAN.R-project.org/package=dlm}.
}

\seealso{
\code{\link[KFKSDS]{KF}};
\code{\link[stsm]{char2numeric}} in package \pkg{stsm}.
}

\examples{
# state space representation of a structural time series model 
# with arbitrary parameter values
require("stsm")
m <- stsm::stsm.model(model = "BSM", y = AirPassengers, transP = "StructTS",
  pars = c("var1" = 30, "var2" = 15, "var3" = 1, "var4" = 12))
ss <- stsm::char2numeric(m)
# value of the likelihood using different interfaces and options
KalmanFilter(y = m@y, ss = ss, KF.version = "KFKSDS", KF.args = list(P0cov = FALSE))$mloglik
KalmanFilter(y = m@y, ss = ss, KF.version = "KFKSDS", KF.args = list(P0cov = TRUE))$mloglik
# 'StructTS' does not include some constants
KalmanFilter(y = m@y, ss = ss, KF.version = "StructTS")$mloglik
}

\keyword{ts, model}
