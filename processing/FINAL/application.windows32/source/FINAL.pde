//Imports Win1
import controlP5.*;
import processing.net.*; 
import uibooster.*;
import uibooster.model.*;
import uibooster.components.*; 
import java.util.Scanner;
import processing.opengl.*;
import guru.ttslib.*;
UiBooster booster;
Client myClient;
Client myClient1;
PImage img;
PImage im3;
PImage imgb;
PImage p;
String y;
String h;
int k;
//Import Win 2

TTS tts;
PImage img10;
PImage img11;
int maxImages = 7; // Total # of images
int imageIndex = 0; // Initial image to be displayed is the first
boolean mouthOpen = false;
boolean speaking = false;
PImage[] images = new PImage[maxImages];
int mouseClicks;
String []t;

PFont f;
PFont f1;
PWindow win;
DepQ win1;
carte win2;
ControlP5 cp5;
Splashscreen splash1;


public void settings() {
     fullScreen();
}
 
void setup() {
myClient = new Client(this, "127.0.0.1", 61915);
myClient1 = new Client(this, "127.0.0.1", 12121); 
 img =loadImage("r2d2.jpg");
p=loadImage("emineslogo2.png");
im3=loadImage("cursor.png");
img10 = loadImage("10.png");
  img11 = loadImage("8.png");
  img1 = loadImage("03.png");
  img2 = loadImage("02.png");
   f = createFont("Georgia", 40);
  f1 = createFont("Arial", 40); 
imgb =loadImage("mask.jpg");
   image(img11, 0, 0,width,height);
  cp5 = new ControlP5(this); 
   booster = new UiBooster();
  for (int i = 0; i < images.length; i ++ ) {
    images[i] = loadImage(  i+".png" );
  }
  frameRate(7);
   tts = new TTS("kevin16");
   
   
}
 
void draw() {
   
  //if (keyPressed) {  if (key == 'b' || key == 'B') {
  //  win = new PWindow(this);
  //  noLoop();
  //  }
      
mouth();
//thread("ttsSpeak");
if (myClient.available() > 0) {
y= myClient.readString();
k=int(y);
 print(y);
if (k==0){
 mouthOpen = true;
}
else if (k==1){
   mouthOpen = false;
}
else if(k==5){
  String splashImage = dataPath("ma.jpg");
  splash1 = booster.showSplashscreen(splashImage);
  delay(1000);
  splash1.hide();}
  else if (k==6){
     String splashImage1 = dataPath("thanks.png");
    Splashscreen splash = booster.showSplashscreen(splashImage1);
    delay(5000);
    splash.hide();
    win = new PWindow(this);
  }}
  if (myClient1.available() > 0) {
win2= new carte(this);
delay(200);
 win2.exit();

  }
}
void mousePressed() {
   if (mouseButton == LEFT) { mouseClicks++; } else { mouseClicks = 0; }
if(mouseClicks==1){
  win = new PWindow(this);
  

  //win2 = new carte(this);
  //delay(100);  
  //win2.exit();
  
//setTimeout(exit(){win2.close()};1000);
//win2.setTimeout("win2.exit()",1000);
}

else{ 
win = new PWindow(this);

}}
  void exit()
  {
    dispose();
    win1=null;
  }



 
 
void mouth(){
    if (mouthOpen && frameCount % 40>0 //if mouthOpen is true, mouth opens every 40 frames and stays open for 15 frames
&& frameCount % 40 <40){ 
 image(img10, 0, 0,width,height);
   image(images[imageIndex], 620, 670,700,600);
   imageIndex = int(random(images.length));}
else{ // if mouthOpen is false, mouth closes
     image(img11, 0, 0,width,height);
     
   }
}



//void ttsSpeak(){ //if speaking is true, speaking returns false, the robot speaks, and then mouthOpen returns false
//if (speaking){
//speaking = false;
//tts.speak("Hi  choose ur language");
//mouthOpen = false;
//}  
//}
