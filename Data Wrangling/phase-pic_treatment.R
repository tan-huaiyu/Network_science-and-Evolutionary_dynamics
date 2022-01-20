mydata<-read.table("clipboard",header = T)
alpha<-c(1:99)/100
lambda<-c(1:99)/100
treatment<-as.data.frame(matrix(nrow = length(alpha),ncol = length(lambda)))
for (i in c(1:length(alpha))) {
  treatment[i,]<-mydata$remain_rate[which(mydata$alpha==alpha[i])]
}

write.csv(treatment,"treatment_data.csv")
