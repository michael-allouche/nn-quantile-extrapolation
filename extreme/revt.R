library(evt0)

# Compute rho and beta from evt0 package
# ======================================
get_rho_beta <- function(x){
    model<- mop(x,p=0,k=1,"RBMOP")
    return(list(model$rho, model$beta))
}


