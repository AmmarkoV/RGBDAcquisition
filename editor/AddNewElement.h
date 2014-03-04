#ifndef ADDNEWELEMENT_H
#define ADDNEWELEMENT_H


#include "../acquisitionSegment/AcquisitionSegment.h"
#include <wx/listctrl.h>

//(*Headers(AddNewElement)
#include <wx/dialog.h>
#include <wx/listctrl.h>
#include <wx/button.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
#include <wx/choice.h>
//*)


double getwxListDouble(wxListCtrl * theList , unsigned int col , unsigned int row );
unsigned int  getwxListInteger(wxListCtrl * theList , unsigned int col , unsigned int row );

class AddNewElement: public wxDialog
{
	public:

		AddNewElement(wxWindow* parent,wxWindowID id=wxID_ANY,const wxPoint& pos=wxDefaultPosition,const wxSize& size=wxDefaultSize);
		virtual ~AddNewElement();

		//(*Declarations(AddNewElement)
		wxChoice* ChoiceHowToAdd;
		wxStaticText* StaticText1;
		wxButton* ButtonAdd;
		wxButton* ButtonCancel;
		wxTextCtrl* TextCtrl1;
		wxStaticText* StaticText2;
		wxListCtrl* ListCtrlCopiedPointList;
		//*)


        struct SegmentationFeaturesDepth * segDepth;
        struct SegmentationFeaturesRGB * segRGB;
        wxListCtrl* ListCtrlPoints;
        unsigned int moduleID;
        unsigned int devID;

	protected:

		//(*Identifiers(AddNewElement)
		static const long ID_BUTTON1;
		static const long ID_STATICTEXT1;
		static const long ID_BUTTON2;
		static const long ID_TEXTCTRL1;
		static const long ID_STATICTEXT2;
		static const long ID_CHOICE1;
		static const long ID_LISTCTRL1;
		//*)

	private:

		//(*Handlers(AddNewElement)
		void OnButtonCancelClick(wxCommandEvent& event);
		void OnButtonAddClick(wxCommandEvent& event);
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
