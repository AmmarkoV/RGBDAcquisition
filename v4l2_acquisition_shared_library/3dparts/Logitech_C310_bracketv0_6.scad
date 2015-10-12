module backFastner()
{ 
   difference()
   {     
        //Draw Back Plate
        color([255,0,0,255])
        {
         translate([0,0,0.2]) { cube([7.7, 4, 0.3]); }
        }
        
        //Screw Center
        //translate([23,20,0]) { cylinder(h=5, r=0.15); }  
        //Screw Down
        //translate([33,5,0]) { cylinder(h=5, r=0.15); }  
   } 

}


module cameraBracket()
{  
   //     Ax            Bx            Cx   Dx   Ex     
   //          _____________________     _____ 
   //       /                        \__/      \  
   //     |                                     |
   //     |                                     |
   //     |   _                        __       |
   //      \_/ \_____________________/   \_____/
   //

   FullLength=5.7;   

   Ax=0.84852813;
   Bx=2.7;
   Cx=0.4;
   Dx=0.9;
   Ex=0.9219544;  
   
   ABx = Ax + Bx;
   ABCx = ABx + Cx;
   ABCDx = ABCx + Dx;
   ABCDEx = ABCDx + Ex;



   Ayl=0.3;
   Byl=1.3; 
   Cyl=0.3236067; 
   AByl = Ayl+Byl;
   ABCyl = AByl+Cyl;



   Ayr=0.6;
   Byr=0.9; 
   Cyr=0.6;

   down = Ayr+Byr+Cyr;
 
   sX = 1.0;
   sY = 1.0; 
 

difference()
 {    

  linear_extrude(height = 0.7,center = true,convexity = 10,twist = 0,slices = 1,scale = 1.0)
  { 
   polygon(
            points=[
                    [sX+0        , sY+Ayl],    // 0 
                    [sX+Ax       , sY],        // 1
                    [sX+ABx      , sY],        // 2
                    [sX+ABx      , sY+0.2],    // 3
                    [sX+ABCx     , sY+0.2],    // 4
                    [sX+ABCx     , sY],        // 5
                    [sX+ABCDx    , sY],        // 6
                    [sX+ABCDEx+0.5   , sY],     // 7 +Ayr for good shape
                    [sX+ABCDEx+0.5   , sY+Ayr+Byr],  // 8 
                    [sX+ABCDx    , sY+down],     // 9
                    [sX+ABCx     , sY+down],     // 10
                    [sX+ABCx     , sY+down-0.2], // 11
                    [sX+ABx      , sY+down-0.2], // 12
                    [sX+ABx      , sY+down],     //13
                    [sX+ABx-2.4  , sY+down],     //14
                    [sX+ABx-2.4  , sY+down-0.9],   //15
                    [sX+ABx-2.8   , sY+down-0.9],     //16
                    [sX+ABx-2.8   , sY+ABCyl],     //17
                    [sX+0.2   , sY+down],     //18
                    [sX-0.5   , sY+down],     //19
                    [sX-0.5   , sY+Ayl],     //20
                    [sX        , sY+Ayl]    // 21 


                   ], paths=[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
         );
  }  
    
  //Remove Space for the plug
  translate([sX,sY,-0.35]) { cube([1.5, 0.8, 0.36]); }
  //Remove Space for Groud Solder
  translate([sX+0.3,sY+1.3,-0.35]) { cube([0.2, 0.2, 0.36]); }


  //Screw Bottom Left
  translate([sX+0.2,sY+1.6,-0.35]) { cylinder(h=5, r=0.10); }  
  //Screw Top Right
  translate([sX+FullLength-0.3,sY+0.54,-0.35]) { cylinder(h=5, r=0.10); }  
}

  /*
 color([0,255,0,255])
        {
         translate([sX,sY+1.6,-0.4]) { cube([0.7, 0.8, 0.4]); }
        }
*/
 
}
  
backFastner();
cameraBracket();








