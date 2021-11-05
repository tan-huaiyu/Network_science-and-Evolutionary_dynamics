mydata<-read.table("clipboard",header = T)

r_lst<-unique(mydata$r)
init<-rep(0,length(mydata$groups[which(mydata$r==2)]))

for (r in r_lst) {
  init<-cbind(init,mydata$groups[which(mydata$r==r)])
}

labels_mat<-init[,-1]
labels_mat<-t(labels_mat)
