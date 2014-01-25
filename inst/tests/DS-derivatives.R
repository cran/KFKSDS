
# tests for disturbance smoother
# compare numerical and analytical derivatives and 
# different funcions to compute the latter; 
# all of them should yield same results up to a tolerance

library("stsm.class")
library("KFKSDS")
library("numDeriv")

m <- stsm.class::stsm.model(model = "llm+seas", y = JohnsonJohnson, 
  pars = c("var1" = 2, "var2" = 15, "var3" = 30))
ss <- stsm.class::char2numeric(m)

convergence <- c(0.001, length(m@y))
#convergence <- c(0.001, 10)

#m@y[10] <- NA # NA before convergence
#m@y[45] <- NA # NA after convergence

kf <- KF(m@y, ss, convergence = convergence)
ks <- KS(m@y, ss, kf)
ds <- DS(m@y, ss, kf, ks)

kfd <- KF.deriv(m@y, ss, convergence = convergence)
ksd <- KS.deriv(m@y, ss, kfd)
dsd <- DS.deriv(ss, ksd)

# derivatives

fcn <- function(x, model, type, i = 1)
{
  m <- stsm.class::set.pars(model, x)
  ss <- stsm.class::char2numeric(m)
  kf <- KF(m@y, ss)
  ks <- KS(m@y, ss, kf)
  ds <- DS(m@y, ss, kf, ks)
  
  switch(type,
    "epshat" = sum(ds$epshat),
    "vareps" = sum(ds$vareps),
    "etahat" = sum(ds$etahat[,i]),
    "vareta" = sum(ds$vareta[,i,i]))
}

d <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "epshat")
all.equal(d, dsd$depshat)

d <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "vareps")
all.equal(d, dsd$dvareps)

for (i in 1:2)
{
d <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "etahat", i = i)
print(all.equal(d, dsd$detahat[i,]))
}

for (i in 1:2)
{
d <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "vareta", i = i)
print(all.equal(d, dsd$dvareta[i,]))
}
