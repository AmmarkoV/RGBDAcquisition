#include "GetExtrinsics.h"

#include "../acquisition/Acquisition.h"

//(*InternalHeaders(GetExtrinsics)
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(GetExtrinsics)
const long GetExtrinsics::ID_STATICTEXT1 = wxNewId();
const long GetExtrinsics::ID_SPINCTRL1 = wxNewId();
const long GetExtrinsics::ID_STATICTEXT2 = wxNewId();
const long GetExtrinsics::ID_SPINCTRL2 = wxNewId();
const long GetExtrinsics::ID_STATICTEXT3 = wxNewId();
const long GetExtrinsics::ID_TEXTCTRL1 = wxNewId();
const long GetExtrinsics::ID_STATICTEXT4 = wxNewId();
const long GetExtrinsics::ID_BUTTON1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(GetExtrinsics,wxDialog)
	//(*EventTable(GetExtrinsics)
	//*)
END_EVENT_TABLE()

GetExtrinsics::GetExtrinsics(wxWindow* parent,wxWindowID id,const wxPoint& pos,const wxSize& size)
{
	//(*Initialize(GetExtrinsics)
	Create(parent, id, _("Get Extrinsics"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxSize(582,118));
	Move(wxDefaultPosition);
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Width"), wxPoint(24,48), wxDefaultSize, 0, _T("ID_STATICTEXT1"));
	Width = new wxSpinCtrl(this, ID_SPINCTRL1, _T("7"), wxPoint(72,40), wxSize(48,27), 0, 3, 100, 7, _T("ID_SPINCTRL1"));
	Width->SetValue(_T("7"));
	StaticText2 = new wxStaticText(this, ID_STATICTEXT2, _("Height"), wxPoint(136,48), wxDefaultSize, 0, _T("ID_STATICTEXT2"));
	Height = new wxSpinCtrl(this, ID_SPINCTRL2, _T("4"), wxPoint(192,40), wxSize(48,27), 0, 3, 100, 4, _T("ID_SPINCTRL2"));
	Height->SetValue(_T("4"));
	StaticText3 = new wxStaticText(this, ID_STATICTEXT3, _("Size"), wxPoint(272,48), wxDefaultSize, 0, _T("ID_STATICTEXT3"));
	Scale = new wxTextCtrl(this, ID_TEXTCTRL1, _("0.03"), wxPoint(304,40), wxSize(48,27), 0, wxDefaultValidator, _T("ID_TEXTCTRL1"));
	StaticText4 = new wxStaticText(this, ID_STATICTEXT4, _("units"), wxPoint(360,48), wxDefaultSize, 0, _T("ID_STATICTEXT4"));
	ButtonGetExtrinsics = new wxButton(this, ID_BUTTON1, _("Get Extrinsics"), wxPoint(416,40), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON1"));

	Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&GetExtrinsics::OnButtonGetExtrinsicsClick);
	//*)
}

GetExtrinsics::~GetExtrinsics()
{
	//(*Destroy(GetExtrinsics)
	//*)
}


void GetExtrinsics::OnButtonGetExtrinsicsClick(wxCommandEvent& event)
{
 char what2run[2048]={0};
 unsigned int w = 7 , h = 4;
 double ds=0;
 float s = 0.0666;

 w=Width->GetValue();
 h=Height->GetValue();

 Scale->GetValue().ToDouble(&ds);
 s= (float) ds;


 unsigned int width , height , channels , bitsperpixel;
 acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);


 acquisitionSaveRawImageToFile((char *) "getExtrinsics.pnm",acquisitionGetColorFrame(moduleID,devID), width , height , channels , bitsperpixel);
 fprintf(stderr,"Width = %u , Height = %u , Scale = %0.5f \n",w,h,s);
 sprintf(what2run,"../tools/ExtrinsicCalibration/extrinsicCalibration -w %u -h %u -s %0.5f --writeImage -i getExtrinsics.pnm -c colorAR.calib --writeImage",w,h,s);


 fprintf(stderr,"What will run %s \n",what2run);

 int i = system(what2run);
 if (i!=0) { fprintf(stderr,"Failed running \n"); }

 ReadCalibration("colorAR.calib",width,height,&calib);

 return;
}
