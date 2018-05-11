#include "SelectModule.h"

#include <wx/msgdlg.h>
#include "../acquisition/Acquisition.h"
#include "../tools/OperatingSystem/OperatingSystem.h"

#define LIST_MAX_LENGTH 32000
#define LISTITEM_MAX_LENGTH 4096

//(*InternalHeaders(SelectModule)
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(SelectModule)
const long SelectModule::ID_BUTTON1 = wxNewId();
const long SelectModule::ID_STATICTEXT1 = wxNewId();
const long SelectModule::ID_COMBOBOX1 = wxNewId();
const long SelectModule::ID_STATICTEXT2 = wxNewId();
const long SelectModule::ID_STATICTEXT3 = wxNewId();
const long SelectModule::ID_TEXTCTRL2 = wxNewId();
const long SelectModule::ID_STATICTEXT4 = wxNewId();
const long SelectModule::ID_TEXTCTRL3 = wxNewId();
const long SelectModule::ID_STATICTEXT5 = wxNewId();
const long SelectModule::ID_TEXTCTRL4 = wxNewId();
const long SelectModule::ID_STATICTEXT6 = wxNewId();
const long SelectModule::ID_BUTTON2 = wxNewId();
const long SelectModule::ID_COMBOBOX2 = wxNewId();
const long SelectModule::ID_CHECKBOX1 = wxNewId();
const long SelectModule::ID_CHECKBOX2 = wxNewId();
//*)

BEGIN_EVENT_TABLE(SelectModule,wxDialog)
	//(*EventTable(SelectModule)
	//*)
END_EVENT_TABLE()

SelectModule::SelectModule(wxWindow* parent,wxWindowID id,const wxPoint& pos,const wxSize& size)
{
	//(*Initialize(SelectModule)
	Create(parent, id, _("Select Used Module"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxSize(297,226));
	Move(wxDefaultPosition);
	ButtonStartModule = new wxButton(this, ID_BUTTON1, _("Start Module"), wxPoint(24,160), wxSize(128,27), 0, wxDefaultValidator, _T("ID_BUTTON1"));
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Module : "), wxPoint(24,22), wxSize(64,24), 0, _T("ID_STATICTEXT1"));
	ComboBoxModule = new wxComboBox(this, ID_COMBOBOX1, wxEmptyString, wxPoint(88,16), wxDefaultSize, 0, 0, 0, wxDefaultValidator, _T("ID_COMBOBOX1"));
	ComboBoxModule->Append(_("NONE"));
	ComboBoxModule->Append(_("V4L2"));
	ComboBoxModule->Append(_("V4L2 STEREO"));
	ComboBoxModule->Append(_("FREENECT"));
	ComboBoxModule->Append(_("OPENNI1"));
	ComboBoxModule->Append(_("OPENNI2"));
	ComboBoxModule->Append(_("OPENGL"));
	ComboBoxModule->SetSelection( ComboBoxModule->Append(_("TEMPLATE")) );
	ComboBoxModule->Append(_("NETWORK"));
	ComboBoxModule->Append(_("DEPTHSENSE"));
	ComboBoxModule->Append(_("REALSENSE"));
	ComboBoxModule->Append(_("DESKTOP"));
	ComboBoxModule->Append(_("SCRIPTED"));
	StaticText2 = new wxStaticText(this, ID_STATICTEXT2, _("Device :"), wxPoint(24,56), wxDefaultSize, 0, _T("ID_STATICTEXT2"));
	StaticText3 = new wxStaticText(this, ID_STATICTEXT3, _("Size"), wxPoint(24,104), wxDefaultSize, 0, _T("ID_STATICTEXT3"));
	TextCtrlWidth = new wxTextCtrl(this, ID_TEXTCTRL2, _("640"), wxPoint(80,100), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL2"));
	StaticText4 = new wxStaticText(this, ID_STATICTEXT4, _("x"), wxPoint(130,104), wxDefaultSize, 0, _T("ID_STATICTEXT4"));
	TextCtrlHeight = new wxTextCtrl(this, ID_TEXTCTRL3, _("480"), wxPoint(144,100), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL3"));
	StaticText5 = new wxStaticText(this, ID_STATICTEXT5, _("@"), wxPoint(192,104), wxDefaultSize, 0, _T("ID_STATICTEXT5"));
	TextCtrlFPS = new wxTextCtrl(this, ID_TEXTCTRL4, _("30"), wxPoint(216,100), wxSize(40,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL4"));
	StaticText6 = new wxStaticText(this, ID_STATICTEXT6, _("fps"), wxPoint(264,104), wxDefaultSize, 0, _T("ID_STATICTEXT6"));
	ButtonCancel = new wxButton(this, ID_BUTTON2, _("Cancel"), wxPoint(184,160), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON2"));
	ComboBoxDevice = new wxComboBox(this, ID_COMBOBOX2, wxEmptyString, wxPoint(88,56), wxDefaultSize, 0, 0, wxCB_SORT|wxTE_PROCESS_ENTER, wxDefaultValidator, _T("ID_COMBOBOX2"));
	ComboBoxDevice->Append(wxEmptyString);
	ComboBoxDevice->Append(_("Plugin Not Availiable"));
	CheckBoxColorStream = new wxCheckBox(this, ID_CHECKBOX1, _("Color Stream"), wxPoint(24,128), wxDefaultSize, 0, wxDefaultValidator, _T("ID_CHECKBOX1"));
	CheckBoxColorStream->SetValue(true);
	CheckBoxDepthStream = new wxCheckBox(this, ID_CHECKBOX2, _("Depth Stream"), wxPoint(160,128), wxDefaultSize, 0, wxDefaultValidator, _T("ID_CHECKBOX2"));
	CheckBoxDepthStream->SetValue(true);

	Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&SelectModule::OnButtonStartModuleClick);
	Connect(ID_COMBOBOX1,wxEVT_COMMAND_COMBOBOX_SELECTED,(wxObjectEventFunction)&SelectModule::OnComboBoxModuleSelected);
	Connect(ID_BUTTON2,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&SelectModule::OnButtonCancelClick);
	Connect(ID_COMBOBOX2,wxEVT_COMMAND_COMBOBOX_SELECTED,(wxObjectEventFunction)&SelectModule::OnComboBoxDeviceSelected);
	//*)


    Connect(ID_COMBOBOX2,wxEVT_COMMAND_ENTER,(wxObjectEventFunction)&SelectModule::OnButtonStartModuleClick);
    Connect(ID_COMBOBOX2,wxEVT_CHAR, (wxObjectEventFunction)&SelectModule::myComboKeyPress);
    Connect(ID_COMBOBOX2,wxEVT_KEY_DOWN,(wxObjectEventFunction)&SelectModule::myComboKeyPress);

  //char listOfFiles[4096];


 moduleSelected=TEMPLATE_ACQUISITION_MODULE;
 deviceSelected=0;
 widthSelected=640;
 heightSelected=480;
 fpsSelected=60;

 ignoreColor=0;
 ignoreDepth=0;

  selectionMade=0;
  ReloadDevicesForSelectedModule();
}

SelectModule::~SelectModule()
{
	//(*Destroy(SelectModule)
	//*)
}


void SelectModule::OnButtonStartModuleClick(wxCommandEvent& event)
{
   long value;
  // DEPTH MAP
  this->moduleSelected = ComboBoxModule->GetCurrentSelection();

  if (ComboBoxDevice->GetValue().ToLong(&value)) {  this->deviceSelected = value; }
  if (TextCtrlWidth->GetValue().ToLong(&value)) {  this->widthSelected = value; }
  if (TextCtrlHeight->GetValue().ToLong(&value)) {  this->heightSelected = value; }
  if (TextCtrlFPS->GetValue().ToLong(&value)) {  this->fpsSelected = value; }


  if ( CheckBoxColorStream->IsChecked() ) { ignoreColor=0; } else { ignoreColor=1; }
  if ( CheckBoxDepthStream->IsChecked() ) { ignoreDepth=0; } else { ignoreDepth=1; }


  wxString mystring = ComboBoxDevice->GetValue();
  strcpy( this->deviceNameSelected ,  mystring.mb_str() );

  selectionMade=1;
  Close();
   /*
		unsigned int deviceSelected;
		unsigned int widthSelected;
		unsigned int heightSelected;
		unsigned int fpsSelected;

		char deviceNameSelected[512];*/
}

void SelectModule::OnButtonCancelClick(wxCommandEvent& event)
{
    Close();
}

void SelectModule::ReloadDevicesForSelectedModule()
{
  unsigned int modID = ComboBoxModule->GetCurrentSelection();
  if ( !acquisitionIsModuleAvailiable(modID) )
  {
      wxMessageBox(wxT("Could not find this plugin on your installation , see stderr for more information"),wxT("Could not detect module"));
  } else
  {
   unsigned int doLoadUnload = 0;

   if ( !acquisitionPluginIsLoaded(modID) ) { doLoadUnload=1; }
   if (doLoadUnload) { acquisitionLoadPlugin(modID); }

         char newListOfDevices[LIST_MAX_LENGTH]={0};
         char * deviceItem = (char *) malloc(LISTITEM_MAX_LENGTH * sizeof(char));
         acquisitionListDevices(modID,0,newListOfDevices,LIST_MAX_LENGTH);

           fprintf(stderr,"List Devices %s\n",newListOfDevices);

         // You can also convert from many encodings by passing the
         // appropriate wxConv... parameter to the constructor
         ComboBoxDevice->Clear();


         int i=0;
         while ( i < 100)
          {
           //fprintf(stderr,"%u\n",i);
           if ( copyDirectoryListItem( i , newListOfDevices , deviceItem , LISTITEM_MAX_LENGTH ) )
           {
               //fprintf(stderr,"%u - %s \n",i,deviceItem);

               wxString wxStrDeviceList(deviceItem, wxConvUTF8);
               ComboBoxDevice->AppendString(wxStrDeviceList);
           } else
           {
               fprintf(stderr,"A total of %u items\n",i);
               break;
           }

            ++i;
          }

          free(deviceItem);

   if (doLoadUnload) { acquisitionUnloadPlugin(modID); }
  }
}

void SelectModule::OnComboBoxDeviceSelected(wxCommandEvent& event)
{

}

void SelectModule::myComboKeyPress( wxKeyEvent& event )
{
   wxCommandEvent event2;
   if ( event.GetKeyCode() == WXK_RETURN ) { OnButtonStartModuleClick(event2); }

   fprintf(stderr,"KeyCode we got was %u \n",event.GetKeyCode());

   event.Skip();
}


void SelectModule::OnComboBoxModuleSelected(wxCommandEvent& event)
{
    unsigned int modID = ComboBoxModule->GetCurrentSelection();

    if (modID==NO_ACQUISITION_MODULE)
    { /*No module selected*/ }
     else
    {
     ReloadDevicesForSelectedModule();
    }
}
