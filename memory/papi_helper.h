#ifndef __papi_helper_hpp
#define __papi_helper_hpp

#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <string>
#include <cstring>
#include <vector>

#ifdef WITH_PAPI
# include <papi.h>
#endif

#define PAPI_CMD( cmd ) \
{ \
   int __retval__ = (cmd); \
   if (__retval__ != PAPI_OK) { \
      fprintf(stderr, "PAPI Error: failed calling %d %s\n", __retval__, # cmd); \
      exit(-1); \
   } \
}

void
papi_get_events( std::vector<int> &papi_events )
{
#ifdef WITH_PAPI

   char *papi_env = getenv("PAPI_EVENTS");
   if (papi_env)
   {
      std::vector< std::string > papi_event_names;

      printf("papi_env %s\n", papi_env);

      char *str = papi_env;
      size_t len = strlen(papi_env);

      while (str < papi_env + len)
      {
         char *next = strchr( str, ',' );
         if (next) {
            papi_event_names.push_back( std::string( str, next-str ) );
            str = next + 1;
         }
         else {
            papi_event_names.push_back( std::string( str ) );
            break;
         }
      }

      papi_events.clear();

      for (int i = 0; i < papi_event_names.size(); ++i)
      {
         int this_event = 0 | PAPI_NATIVE_MASK;
         int retval = PAPI_event_name_to_code( const_cast<char*>( papi_event_names[i].c_str() ), &this_event );
         if (retval != PAPI_OK) {
            fprintf(stderr,"PAPI: Error calling PAPI_event_code_to_name %d\n", retval);
            exit(-1);
         }
         printf("PAPI event %s %x\n", papi_event_names[i].c_str(), this_event);
         papi_events.push_back( this_event );
      }
   }
#endif

   return;
}

int papi_start(void)
{
#ifdef WITH_PAPI
   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
      fprintf(stderr,"PAPI: version mismatch!\n");
      return 1;
   }
   if (PAPI_num_counters() < 2) {
      fprintf(stderr,"PAPI: no hardware counters available!\n");
      return 1;
   }
#endif
   return 0;
}

int papi_stop(void)
{
#ifdef WITH_PAPI
   PAPI_shutdown();
#endif

   return 0;
}

#endif
