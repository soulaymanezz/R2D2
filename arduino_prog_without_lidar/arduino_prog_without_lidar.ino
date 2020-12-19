#include <FlexiTimer2.h>

/* Ce code source a pour but principal la commande du servomoteur, du motoréducteur aux deux sens en utilisant deux touches d'accélération et de décélaration, puis
 *  d'arrêter la machine avec la touche 'S'.
 */



#include <digitalWriteFast.h>
 int kp=10,kd=0,ki=0;   //1 2
 int g=1;
#define dir1 9
#define pwm1 6
#define encod1b 21
#define interrupt1b 2
#define encod1a 20
#define interrupt1a 3

#define dir2 4                            
#define pwm2 7
#define encod2b 19
#define interrupt2b 4
#define encod2a 18
#define interrupt2a 5

#define dir3 5
#define pwm3 8
#define encod3b 2
#define interrupt3b 0
#define encod3a 3
#define interrupt3a 1
char commande;
float x=0.0,teta=0.0,y=0.0;
float commande1=0,commande2=0,commande3=0;
float PWM1=0,PWM2=0,PWM3=0;
int PWM = 240,cnt=2;
float R=0.27,r=0.05;
volatile long compteurImpulsions1 = 1,c1=1;
volatile long compteurImpulsions2 = 1,c2=1;
volatile long compteurImpulsions3 = 1,c3=1;
// Pour le moteur à courant continu:
unsigned long time;
//variables for serial read and communication
String string=" ";
int lenInt;
int bin=1;
String D[30];
float dv[30];

float alpha1=0.0,alpha2=0.0,alpha_m=0.0,alpha3=0.0;

float err1=0,err2=0,err3=0;
float errp1=0,errp2=0,errp3=0;
float sumerr1=0,sumerr2=0,sumerr3=0;


void setup(void)

  {
  // Pour le codeur incrémental
  pinMode(encod1a,INPUT_PULLUP); pinMode(encod1b,INPUT_PULLUP);
pinMode(encod2a,INPUT_PULLUP); pinMode(encod2b,INPUT_PULLUP);
pinMode(encod3a,INPUT_PULLUP); pinMode(encod3b,INPUT_PULLUP);

 

pinMode(pwm1,OUTPUT); pinMode(pwm2,OUTPUT); pinMode(pwm3,OUTPUT);
pinMode(dir1,OUTPUT); pinMode(dir2,OUTPUT); pinMode(dir3,OUTPUT);
Serial.begin(9600);
  attachInterrupt(interrupt1a, GestionInterruptionCodeurPin1A, CHANGE);
  attachInterrupt(interrupt1b, GestionInterruptionCodeurPin1B, CHANGE);
   attachInterrupt(interrupt2a, GestionInterruptionCodeurPin2A, CHANGE);
  attachInterrupt(interrupt2b, GestionInterruptionCodeurPin2B, CHANGE);
   attachInterrupt(interrupt3a, GestionInterruptionCodeurPin3A, CHANGE);
  attachInterrupt(interrupt3b, GestionInterruptionCodeurPin3B, CHANGE);

  // Pour compteur d'impulsions de l'encodeur:

  
 //FlexiTimer2::set(100, 1/1000., is);
 //FlexiTimer2::start();

  }



void loop()
    {
Serial.flush();
if(Serial.available() >0) {
  string = Serial.readStringUntil('\r\n');
  
  c1=1;c2=1;c3=1;
  bin=0;}

if(string!=" "){
   String len = getValue(string, ';', 0);
   lenInt=len.toInt();
   if (lenInt!=0){
       for (int i=1;i<lenInt;i=i+2){
       D[(i-1)/2]=getValue(string, ';', i);
 
       String xval = getValue(string, ';', i+1);
       dv[(i-1)/2]=xval.toFloat();
      
  
   } }
   string=" ";}
if (bin==0){
   for (int i=0;i<lenInt/2;i++){
     
      if(D[i]=="RR"){
          x=-dv[i];
          y=0;
         }
      if(D[i]=="LL"){
          x=dv[i];
          y=0;}
      if(D[i]=="DD"){
          y=-dv[i];
          x=0;}
      if(D[i]=="UU"){
          y=dv[i];
          x=0;}
      if(D[i]=="RD"){
          x=-cos(PI/4)*dv[i];
          y=+cos(PI/4)*dv[i];
         
          }
      if(D[i]=="RU"){
          x=-cos(PI/4)*dv[i];
          y=(cos(PI/4))*dv[i];}
      if(D[i]=="LD"){
          x=(cos(PI/4))*dv[i];
          y=-cos(PI/4)*dv[i];}
      if(D[i]=="LU"){
          x=(cos(PI/4))*dv[i];
          y=(cos(PI/4))*dv[i];}
          teta=0.0;
        g=1;
      equations(x,y,teta);
    }
   
    PWM1=0;PWM2=0;PWM3=0;
  
   pwm_cond(PWM1,PWM2,PWM3);
   
   
    while(g==1)
     {if (Serial.available()>0){
      g=Serial.parseInt();
     }
      Serial.println("ok");
      
     }
    bin=1;
 }}
 
 













void equations(float x,float y ,float teta){
alpha1=0.0;   alpha2=0.0;  alpha3=0.0;  alpha_m=0.0;
commande3=(y+R*teta)/(2*PI*r)  ;
commande2= (-sin(4*PI/3)*x+cos(4*PI/3)*y+R*teta)/(2*PI*r); //vx en cm /s vteta tr/s
commande1=(-sin(2*PI/3)*x+cos(2*PI/3)*y+R*teta)/(2*PI*r);
//Serial.print(commande1);Serial.print("ccccc");Serial.print(commande2);Serial.print("ccccc");
//Serial.print(commande3);Serial.println("ccccc");

compteurImpulsions1=commande1;  compteurImpulsions2=commande2;  compteurImpulsions3=commande3;
float maxx=max(max(abs(commande1),abs(commande2)),abs(commande3));

PWM1=PWM*commande1/maxx;PWM2=PWM*commande2/maxx;PWM3=PWM*commande3/maxx;

  go();

}











void go() {
 
if(commande1!=0.0){alpha1=abs(compteurImpulsions1)/(commande1*2080);}
if(commande2!=0.0){alpha2=abs(compteurImpulsions2)/(commande2*2080);}
if(commande3!=0.0){alpha3=abs(compteurImpulsions3)/(commande3*2080);}

while(abs(alpha1)<=1.0 && abs(alpha2)<=1.0 && abs(alpha3)<=1.0){
 
if(commande1!=0.0){alpha1=abs(compteurImpulsions1)/(commande1*2080);}
if(commande2!=0.0){alpha2=abs(compteurImpulsions2)/(commande2*2080);}
if(commande3!=0.0){alpha3=abs(compteurImpulsions3)/(commande3*2080);}


if(commande1!=0.0 && commande2!=0.0 &&commande3!=0.0){alpha_m=(abs(alpha2)+abs(alpha1)+abs(alpha3))/3.0;}
else if(commande1==0.0){alpha_m=(abs(alpha2)+abs(alpha3))/2.0;}
else if(commande2==0.0){alpha_m=(abs(alpha1)+abs(alpha3))/2.0;}
else if(commande3==0.0){alpha_m=(abs(alpha2)+abs(alpha1))/2.0;}
err1=(alpha_m-abs(alpha1));
err2=(alpha_m-abs(alpha2));
err3=(alpha_m-abs(alpha3));
sumerr1+=err1;if (abs(err1)<0.01){sumerr1=0;}
sumerr2+=err2;if (abs(err2)<0.01){sumerr2=0;}
sumerr3+=err3;if (abs(err3)<0.01){sumerr3=0;}
//Serial.print(compteurImpulsions1);Serial.print("********");Serial.print(compteurImpulsions2);Serial.print("********");Serial.print(compteurImpulsions3);Serial.println("********");
//Serial.print(alpha1);Serial.print("********");Serial.print(alpha2);Serial.print("********");Serial.print(alpha3);Serial.println("********");
if(abs(alpha1)<=1.0 && commande1>0.0){PWM1+=kp*err1+kd*(err1-errp1)+ki*err1*sumerr1;} else if(abs(alpha1)<=1.0 && commande1<0.0){PWM1-=kp*err1+kd*(err1-errp1)+ki*err1*sumerr1;}
if(abs(alpha2)<=1.0 && commande2>0.0){PWM2+=kp*err2+kd*(err2-errp2)+ki*err2*sumerr2;} else if(abs(alpha2)<=1.0 && commande2<0.0){PWM2-=kp*err2+kd*(err2-errp2)+ki*err2*sumerr2;}
if(abs(alpha3)<=1.0 && commande3>0.0){PWM3+=kp*err3+kd*(err3-errp3)+ki*err3*sumerr3;} else if(abs(alpha3)<=1.0 && commande3<0.0){PWM3-=kp*err3+kd*(err3-errp3)+ki*err3*sumerr3;}
errp1=err1;
errp2=err2;
errp3=err3;
delay(10);
pwm_cond(PWM1,PWM2,PWM3);}

}





































 
void pwm_cond(float PWM1,float PWM2,float PWM3){
if(PWM1>0) {
  digitalWrite(dir1,LOW);
 }
 else if(PWM1<=0) {
  digitalWrite(dir1,HIGH);
 }
  if(PWM2>0) {
  digitalWrite(dir2,LOW);
 }
 else if(PWM2<=0) {
  digitalWrite(dir2,HIGH);
 }
  if(PWM3>0) {
  digitalWrite(dir3,LOW);
 }
 else if(PWM3<=0) {
  digitalWrite(dir3,HIGH);
 }
 if(PWM1>255) {
  PWM1 = 255;
 }
  if(PWM2>255) {
  PWM2 = 255;
 }
  if(PWM3>255) {
  PWM3 = 255;
 }
 if(PWM1<-255) {
  PWM1 = -255;
 }
  if(PWM2<-255) {
  PWM2 = -255;
 }
  if(PWM3<-255) {
  PWM3 = -255;
 }
 int pwm11=(int)PWM1;
 int pwm22=(int)PWM2;
 int pwm33=(int)PWM3;

 analogWrite(pwm1,abs(pwm11));
analogWrite(pwm2,abs(pwm22));

analogWrite(pwm3,abs(pwm33));
}



void is(){


}






void stopp() {
   analogWrite(pwm1,0);
analogWrite(pwm2,0);

analogWrite(pwm3,abs(0));
//Serial.print(alpha1);
//Serial.print("aaaa");
//Serial.print(alpha2);
//Serial.print("aaaaa");
//Serial.println(alpha3);
// Serial.print(PWM1);
// Serial.print("******");
//  Serial.print(PWM2);
// Serial.print("******"); Serial.println(PWM3);
}





   





   
// Pour la routine de service d'interruption attachée à la voie A du codeur incrémental:
void GestionInterruptionCodeurPin1A()
    {
     if (digitalReadFast2(encod1a) == digitalReadFast2(encod1b))
          {compteurImpulsions3++;c3++;}
     else {compteurImpulsions3--;c3--;}
    }
 
// Pour la routine de service d'interruption attachée à la voie B du codeur incrémental:
void GestionInterruptionCodeurPin1B()
    {
     if (digitalReadFast2(encod1a) == digitalReadFast2(encod1b))
          {compteurImpulsions3--;c3--;}
     else {compteurImpulsions3++;c3++;}
    }



void GestionInterruptionCodeurPin2A()
    {
     if (digitalReadFast2(encod2a) == digitalReadFast2(encod2b))
          {compteurImpulsions1++;c1++;}
     else {compteurImpulsions1--;c1--;}
    }
 
// Pour la routine de service d'interruption attachée à la voie B du codeur incrémental:
void GestionInterruptionCodeurPin2B()
    {
     if (digitalReadFast2(encod2a) == digitalReadFast2(encod2b))
          {compteurImpulsions1--;c1--;}
     else {compteurImpulsions1++;c1++;}
    }
    void GestionInterruptionCodeurPin3A()
    {
     if (digitalReadFast2(encod3a) == digitalReadFast2(encod3b))
          {compteurImpulsions2++;c2++;}
     else {compteurImpulsions2--;c2--;}
    }
 
// Pour la routine de service d'interruption attachée à la voie B du codeur incrémental:
void GestionInterruptionCodeurPin3B()
    {
     if (digitalReadFast2(encod3a) == digitalReadFast2(encod3b))
          {compteurImpulsions2--;c2--;}
     else {compteurImpulsions2++;c2++;}
    }



    String getValue(String data, char separator, int index)
{
    int found = 0;
    int strIndex[] = { 0, -1 };
    int maxIndex = data.length() - 1;

    for (int i = 0; i <= maxIndex && found <= index; i++) {
        if (data.charAt(i) == separator || i == maxIndex) {
            found++;
            strIndex[0] = strIndex[1] + 1;
            strIndex[1] = (i == maxIndex) ? i+1 : i;}}
    return found > index ? data.substring(strIndex[0], strIndex[1]) : "";
}
