#ifndef SCANHUMAN_H
#define SCANHUMAN_H

//(*Headers(ScanHuman)
#include <wx/gauge.h>
#include <wx/dialog.h>
#include <wx/button.h>
#include <wx/statbox.h>
#include <wx/textctrl.h>
//*)

class ScanHuman: public wxDialog
{
	public:

		ScanHuman(wxWindow* parent,wxWindowID id=wxID_ANY,const wxPoint& pos=wxDefaultPosition,const wxSize& size=wxDefaultSize);
		virtual ~ScanHuman();

		//(*Declarations(ScanHuman)
		wxTextCtrl* TextCtrlDataset;
		wxButton* ButtonCapture;
		wxStaticBox* StaticBox1;
		wxGauge* Progress;
		wxButton* ButtonRestart;
		//*)

	protected:

		//(*Identifiers(ScanHuman)
		static const long ID_GAUGE1;
		static const long ID_STATICBOX1;
		static const long ID_BUTTON1;
		static const long ID_TEXTCTRL1;
		static const long ID_BUTTON2;
		//*)

	private:

		//(*Handlers(ScanHuman)
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
