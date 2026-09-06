#property service
#property strict
#property version "1.00"
#property description "Task018F: local economic-calendar observations only"

// No includes or DLL imports. No strategy or execution responsibilities.
// API/clock/publication limitations: docs/task018f_bridge.md.
input string InstanceId = "fivepercent-26520700-shadow";
input long ExpectedLogin = 26520700;
input string ExpectedServer = "FivePercentOnline-Real";
input string ExpectedTerminalPath = "C:\\MT5-5ers";
input string ExpectedDataPath = ""; // must be explicitly verified before future start
input int RefreshSeconds = 60;
input int EvidenceExpirySeconds = 90;
input int WindowSeconds = 300;

const int MAX_QUERY_MS = 15000;
const int MAX_EVENTS = 5000;
const int MAX_PAYLOAD_BYTES = 1048576;
const int MAX_QUOTE_AGE = 10;
const int CLOCK_UNCERTAINTY = 2;
const int MAX_GENERATIONS = 1440; // bounded disk use: stop publication, never delete arbitrary files

struct ClockEvidence
  {
   datetime utc;
   datetime server;
   int offset;
   int quote_age;
   bool valid;
  };

string BootId, Directory;
ulong Sequence = 0;
datetime LastUtc = 0, LastQuote = 0;
ulong LastSampleMs = 0, QuoteChangedMs = 0, FaultUntilMs = 0;
int PreviousOffset = 0;
bool HaveOffset = false;

string J(const string value)
  {
   string result = "\"";
   for(int i=0; i<StringLen(value); i++)
     {
      ushort ch = StringGetCharacter(value,i);
      if(ch == 34) result += "\\\"";
      else if(ch == 92) result += "\\\\";
      else if(ch < 32) result += StringFormat("\\u%04x",(int)ch);
      else result += ShortToString(ch);
     }
   return result+"\"";
  }

string N(const long value) { return StringFormat("%I64d",value); }
string U(const ulong value) { return StringFormat("%I64u",value); }
string B(const bool value) { return value ? "true" : "false"; }
string NormalizePath(string value)
  {
   StringReplace(value,"/","\\");
   StringToLower(value);
   while(StringLen(value)>3 && StringSubstr(value,StringLen(value)-1)=="\\")
      value=StringSubstr(value,0,StringLen(value)-1);
   return value;
  }

bool SafeToken(const string value)
  {
   if(StringLen(value)<8 || StringLen(value)>64) return false;
   for(int i=0; i<StringLen(value); i++)
     {
      ushort c=StringGetCharacter(value,i);
      if(!((c>=65 && c<=90)||(c>=97 && c<=122)||(c>=48 && c<=57)||c==45||c==95))
         return false;
     }
   return true;
  }

bool IdentityMatches()
  {
   return AccountInfoInteger(ACCOUNT_LOGIN)==ExpectedLogin
      && AccountInfoString(ACCOUNT_SERVER)==ExpectedServer
      && StringLen(AccountInfoString(ACCOUNT_COMPANY))>0
      && NormalizePath(TerminalInfoString(TERMINAL_PATH))==NormalizePath(ExpectedTerminalPath)
      && NormalizePath(TerminalInfoString(TERMINAL_DATA_PATH))==NormalizePath(ExpectedDataPath);
  }

void ObserveClock(ClockEvidence &sample)
  {
   ulong ms=GetTickCount64();
   sample.utc=TimeGMT();
   sample.server=TimeTradeServer();
   datetime quote=TimeCurrent();
   if(LastSampleMs>0)
     {
      double elapsed=(double)(ms-LastSampleMs)/1000.0;
      if(MathAbs((double)(sample.utc-LastUtc)-elapsed)>CLOCK_UNCERTAINTY)
         FaultUntilMs=ms+(ulong)EvidenceExpirySeconds*1000;
     }
   if(LastQuote>0 && quote>LastQuote) QuoteChangedMs=ms;
   if(LastQuote>0 && quote<LastQuote)
     {
      QuoteChangedMs=0;
      FaultUntilMs=ms+(ulong)EvidenceExpirySeconds*1000;
     }
   double difference=(double)(sample.server-sample.utc);
   sample.offset=(int)MathRound(difference/900.0)*900;
   if(HaveOffset && sample.offset!=PreviousOffset)
      FaultUntilMs=ms+(ulong)EvidenceExpirySeconds*1000;
   sample.quote_age=QuoteChangedMs==0 ? 999999 : (int)((ms-QuoteChangedMs)/1000);
   sample.valid=TerminalInfoInteger(TERMINAL_CONNECTED)!=0
      && quote>0 && sample.utc>0 && sample.server>0
      && sample.offset>=-43200 && sample.offset<=50400
      && MathAbs(difference-sample.offset)<=CLOCK_UNCERTAINTY
      && sample.quote_age<=MAX_QUOTE_AGE
      && MathAbs((double)(sample.server-quote))<=MAX_QUOTE_AGE
      && ms>=FaultUntilMs;
   PreviousOffset=sample.offset;
   HaveOffset=true;
   LastQuote=quote;
   LastUtc=sample.utc;
   LastSampleMs=ms;
  }

string Importance(const ENUM_CALENDAR_EVENT_IMPORTANCE importance)
  {
   switch(importance)
     {
      case CALENDAR_IMPORTANCE_HIGH: return "HIGH";
      case CALENDAR_IMPORTANCE_MODERATE: return "MODERATE";
      case CALENDAR_IMPORTANCE_LOW: return "LOW";
      case CALENDAR_IMPORTANCE_NONE: return "NONE";
     }
   return "UNKNOWN";
  }

string TimeMode(const ENUM_CALENDAR_EVENT_TIMEMODE mode)
  {
   switch(mode)
     {
      case CALENDAR_TIMEMODE_DATETIME: return "DATETIME";
      case CALENDAR_TIMEMODE_DATE: return "DATE";
      case CALENDAR_TIMEMODE_NOTIME: return "NOTIME";
      case CALENDAR_TIMEMODE_TENTATIVE: return "TENTATIVE";
     }
   return "UNKNOWN";
  }

string IdentityJson()
  {
   return "{\"login\":"+J(N(AccountInfoInteger(ACCOUNT_LOGIN)))
      +",\"server\":"+J(AccountInfoString(ACCOUNT_SERVER))
      +",\"company\":"+J(AccountInfoString(ACCOUNT_COMPANY))
      +",\"terminal_path\":"+J(TerminalInfoString(TERMINAL_PATH))
      +",\"terminal_data_path\":"+J(TerminalInfoString(TERMINAL_DATA_PATH))+"}";
  }

bool CurrencyCatalog(int &error)
  {
   MqlCalendarCountry countries[];
   ResetLastError();
   int count=CalendarCountries(countries);
   error=GetLastError();
   if(count<=0 || error!=0 || count!=ArraySize(countries)) return false;
   string required[]={"AUD","CAD","EUR","GBP","JPY","USD"};
   for(int j=0; j<ArraySize(required); j++)
     {
      bool found=false;
      for(int i=0; i<count; i++)
         if(countries[i].currency==required[j]) found=true;
      if(!found) return false;
     }
   return true;
  }

bool ChangeId(ulong &change,int &error)
  {
   MqlCalendarValue unused[];
   change=0;
   ResetLastError();
   int count=CalendarValueLast(change,unused,NULL,NULL);
   error=GetLastError();
   // With input change=0, zero is initialization, NOT an empty calendar.
   return count==0 && error==0 && change>0;
  }

string BuildGeneration()
  {
   ClockEvidence before,after;
   ObserveClock(before);
   ulong start_ms=GetTickCount64();
   string identity=IdentityJson();
   bool identity_ok=IdentityMatches();
   datetime left=before.server-WindowSeconds-CLOCK_UNCERTAINTY;
   datetime right=before.server+WindowSeconds+EvidenceExpirySeconds+CLOCK_UNCERTAINTY;
   datetime from=(datetime)(((long)left/86400)*86400-1);
   datetime to=(datetime)(((long)right/86400+1)*86400+1);
   int count=-1,error=0,change_error_before=0,change_error_after=0,serialized=0;
   ulong change_before=0,change_after=0;
   bool event_complete=true,country_complete=true,catalog_valid=false,query_ok=false;
   string failure="",rows="";
   if(!identity_ok) failure="identity";
   else if(!before.valid) failure="clock_before";
   else if(!CurrencyCatalog(error)) failure="currency_catalog";
   else
     {
      catalog_valid=true;
      if(!ChangeId(change_before,change_error_before)) failure="change_before";
     }
   MqlCalendarValue values[];
   if(failure=="")
     {
      ResetLastError();
      count=CalendarValueHistory(values,from,to,NULL,NULL);
      error=GetLastError();
      // Explicitly reject all errors, including 5400 / partial array results.
      query_ok=count>=0 && error==0 && count==ArraySize(values) && count<=MAX_EVENTS;
      if(!query_ok) failure=error==5400 ? "truncated" : "query";
     }
   if(query_ok)
     {
      for(int i=0; i<count; i++)
        {
         if(IsStopped() || GetTickCount64()-start_ms>(ulong)MAX_QUERY_MS)
           { failure="elapsed"; event_complete=false; break; }
         MqlCalendarEvent event;
         ResetLastError();
         bool event_ok=CalendarEventById(values[i].event_id,event);
         int event_error=GetLastError();
         if(!event_ok || event_error!=0 || event.id!=values[i].event_id)
           { error=event_error; failure="event_enrichment"; event_complete=false; break; }
         MqlCalendarCountry country;
         ResetLastError();
         bool country_ok=CalendarCountryById((long)event.country_id,country);
         int country_error=GetLastError();
         if(!country_ok || country_error!=0 || country.id!=event.country_id)
           { error=country_error; failure="country_enrichment"; country_complete=false; break; }
         if(values[i].id==0 || event.id==0 || country.id==0 || StringLen(event.name)==0
            || StringLen(event.name)>512 || StringLen(country.code)!=2 || StringLen(country.currency)!=3)
           { failure="malformed_enrichment"; event_complete=false; country_complete=false; break; }
         string mode=TimeMode(event.time_mode);
         string utc=mode=="DATETIME" ? N((long)values[i].time-before.offset) : "null";
         string row="{\"value_id\":"+J(U(values[i].id))+",\"event_id\":"+J(U(values[i].event_id))
            +",\"country_id\":"+J(U(event.country_id))+",\"country_code\":"+J(country.code)
            +",\"currency\":"+J(country.currency)+",\"importance\":"+J(Importance(event.importance))
            +",\"time_mode\":"+J(mode)+",\"name\":"+J(event.name)
            +",\"server_time\":"+N((long)values[i].time)+",\"utc_time\":"+utc+"}";
         if(StringLen(rows)+StringLen(row)>MAX_PAYLOAD_BYTES/4)
           { failure="payload_limit"; event_complete=false; break; }
         if(serialized>0) rows+=",";
         rows+=row;
         serialized++;
        }
      if(!ChangeId(change_after,change_error_after) || change_before!=change_after)
         failure="change_after";
     }
   ObserveClock(after);
   long elapsed=(long)(GetTickCount64()-start_ms);
   bool clock_ok=before.valid && after.valid && before.offset==after.offset
      && MathAbs((double)(after.utc-before.utc)*1000.0-elapsed)<=2000;
   if(!clock_ok) failure="clock_after";
   if(elapsed>MAX_QUERY_MS) failure="elapsed";
   bool connected=TerminalInfoInteger(TERMINAL_CONNECTED)!=0;
   if(!connected) failure="disconnected";
   if(!IdentityMatches() || identity!=IdentityJson()) failure="identity_after";
   string clock="{\"generated_server_time\":"+N((long)after.server)
      +",\"generated_utc_time\":"+N((long)after.utc)
      +",\"server_utc_offset_seconds\":"+N(before.offset)
      +",\"offset_sample_time\":"+N((long)after.utc)
      +",\"clock_status\":"+J(clock_ok ? "VALID" : "UNKNOWN")
      +",\"clock_uncertainty_seconds\":"+N(CLOCK_UNCERTAINTY)
      +",\"offset_before_seconds\":"+N(before.offset)+",\"offset_after_seconds\":"+N(after.offset)
      +",\"quote_age_before_seconds\":"+N(before.quote_age)+",\"quote_age_after_seconds\":"+N(after.quote_age)+"}";
   string query="{\"server_start\":"+N((long)from)+",\"server_end\":"+N((long)to)
      +",\"utc_start\":"+N((long)from-before.offset)+",\"utc_end\":"+N((long)to-before.offset)
      +",\"started_utc\":"+N((long)before.utc)+",\"elapsed_ms\":"+N(elapsed)
      +",\"return_count\":"+N(count)+",\"error_code\":"+N(error)
      +",\"query_success\":"+B(query_ok && failure=="")+",\"failure_stage\":"+J(failure)+"}";
   string health="{\"terminal_connected\":"+B(connected)
      +",\"event_enrichment_complete\":"+B(event_complete && serialized==count)
      +",\"country_enrichment_complete\":"+B(country_complete && serialized==count)
      +",\"currency_catalog_valid\":"+B(catalog_valid)
      +",\"change_before\":"+J(U(change_before))+",\"change_after\":"+J(U(change_after))
      +",\"change_error_before\":"+N(change_error_before)+",\"change_error_after\":"+N(change_error_after)+"}";
   string coverage="{\"utc_start\":"+N((long)from-before.offset)+",\"utc_end\":"+N((long)to-before.offset)
      +",\"supported_currencies\":[\"AUD\",\"CAD\",\"EUR\",\"GBP\",\"JPY\",\"USD\"]"
      +",\"returned_event_count\":"+N(serialized)+"}";
   return "{\"schema_version\":1,\"source\":\"mql5-calendar-shadow\",\"instance_id\":"+J(InstanceId)
      +",\"boot_id\":"+J(BootId)+",\"sequence\":"+J(U(Sequence))
      +",\"identity\":"+identity+",\"clock\":"+clock+",\"query\":"+query
      +",\"health\":"+health+",\"coverage\":"+coverage+",\"events\":["+rows+"]}";
  }

bool Utf8(const string text,uchar &bytes[])
  {
   int count=StringToCharArray(text,bytes,0,WHOLE_ARRAY,CP_UTF8);
   if(count<=1) return false;
   // StringToCharArray includes the terminating zero; JSON/digest must not.
   return ArrayResize(bytes,count-1)==count-1;
  }

bool WriteClosed(const string name,const uchar &bytes[])
  {
   ResetLastError();
   int handle=FileOpen(name,FILE_WRITE|FILE_BIN);
   int error=GetLastError();
   if(handle==INVALID_HANDLE || error!=0)
     { if(handle!=INVALID_HANDLE) FileClose(handle); return false; }
   ResetLastError();
   uint written=FileWriteArray(handle,bytes,0,ArraySize(bytes));
   error=GetLastError();
   ResetLastError();
   FileFlush(handle);
   int flush_error=GetLastError();
   ResetLastError();
   FileClose(handle);
   int close_error=GetLastError();
   return written==(uint)ArraySize(bytes) && error==0 && flush_error==0 && close_error==0;
  }

bool Publish(const string payload)
  {
   uchar bytes[],hash[],key[];
   if(!Utf8(payload,bytes) || ArraySize(bytes)>MAX_PAYLOAD_BYTES) return false;
   ResetLastError();
   int hash_size=CryptEncode(CRYPT_HASH_SHA256,bytes,key,hash);
   int hash_error=GetLastError();
   if(hash_size!=32 || hash_error!=0) return false;
   string digest="";
   for(int i=0; i<hash_size; i++) digest+=StringFormat("%02x",(int)hash[i]);
   string filename="calendar_"+BootId+"_"+U(Sequence)+".json";
   string target=Directory+filename;
   string temporary=target+".tmp";
   // No replacement of any generation, including a colliding boot identifier.
   if(FileIsExist(target) || FileIsExist(temporary)) return false;
   if(!WriteClosed(temporary,bytes)) return false;
   ResetLastError();
   bool moved=FileMove(temporary,0,target,0);
   int move_error=GetLastError();
   if(!moved || move_error!=0) return false;
   string manifest="{\"schema_version\":1,\"instance_id\":"+J(InstanceId)
      +",\"boot_id\":"+J(BootId)+",\"sequence\":"+J(U(Sequence))
      +",\"payload_filename\":"+J(filename)+",\"payload_bytes\":"+N(ArraySize(bytes))
      +",\"payload_sha256\":"+J(digest)+",\"published_utc\":"+N((long)TimeGMT())+"}";
   uchar manifest_bytes[];
   if(!Utf8(manifest,manifest_bytes)) return false;
   string manifest_temporary=Directory+"manifest_"+BootId+"_"+U(Sequence)+".tmp";
   if(FileIsExist(manifest_temporary) || !WriteClosed(manifest_temporary,manifest_bytes)) return false;
   // FileMove is NOT assumed atomic. Reader verifies a complete manifest,
   // exact payload bytes/hash and rereads the manifest before accepting.
   ResetLastError();
   moved=FileMove(manifest_temporary,0,Directory+"manifest.json",FILE_REWRITE);
   move_error=GetLastError();
   return moved && move_error==0;
  }

void OnStart()
  {
   if(!SafeToken(InstanceId) || ExpectedDataPath=="" || !IdentityMatches()
      || RefreshSeconds<60 || RefreshSeconds>3600
      || EvidenceExpirySeconds<1 || EvidenceExpirySeconds>90
      || WindowSeconds<1 || WindowSeconds>3600)
     { Print("Calendar shadow service: invalid inputs or pinned identity mismatch"); return; }
   Directory="CalendarBridge\\"+InstanceId+"\\";
   BootId=U((ulong)TimeGMT())+"-"+U(GetMicrosecondCount())+"-"+U(GetTickCount64());
   if(!SafeToken(BootId)) return;
   // Exclusive handle prevents duplicate writers for this instance. Never use
   // FILE_COMMON: output belongs only to this verified terminal's file sandbox.
   int writer_lock=FileOpen(Directory+"writer.lock",FILE_READ|FILE_WRITE|FILE_BIN);
   if(writer_lock==INVALID_HANDLE)
     { Print("Calendar shadow service: writer lock unavailable"); return; }
   ulong next=0;
   while(!IsStopped() && Sequence<(ulong)MAX_GENERATIONS)
     {
      ClockEvidence sample;
      ObserveClock(sample);
      if(GetTickCount64()>=next)
        {
         Sequence++;
         string payload=BuildGeneration();
         if(!Publish(payload)) Print("Calendar shadow publication failed; previous evidence must expire");
         next=GetTickCount64()+(ulong)RefreshSeconds*1000;
        }
      Sleep(1000); // clock monitoring only; calendar calls at refresh cadence
     }
   FileClose(writer_lock);
   Print("Calendar shadow service stopped; evidence expires without refresh");
  }
