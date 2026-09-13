i am going with lite llm appoarch
INFO:     127.0.0.1:61934 - "HEAD /api/hello HTTP/1.1" 404 Not Found
INFO:     127.0.0.1:61942 - "POST /v1/messages?beta=true HTTP/1.1" 200 OK
15:25:42 - LiteLLM Proxy:ERROR: common_request_processing.py:3480 - litellm.proxy.proxy_server.async_data_generator(): Exception occured - litellm.BadRequestError: Vertex_ai_betaException BadRequestError - b'{\n  "error": {\n    "code": 400,\n    "message": "The referenced name `#/components/schemas/HTTPValidationError` in function_response.response does not match to a display_name in the function_response.parts.",\n    "status": "INVALID_ARGUMENT"\n  }\n}\n'
Traceback (most recent call last):
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\llms\vertex_ai\gemini\vertex_and_google_ai_studio_gemini.py", line 2573, in make_call
    response: Final = await client.post(api_base, headers=headers, data=data, stream=True, logging_obj=logging_obj)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\logging_utils.py", line 300, in async_wrapper
    result: Final = await func(*args, **kwargs)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\llms\custom_httpx\http_handler.py", line 721, in post
    await _raise_masked_async_error(e, stream)
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\llms\custom_httpx\http_handler.py", line 480, in _raise_masked_async_error
    raise MaskedHTTPStatusError(e, message=_body, text=_body) from None
litellm.llms.custom_httpx.http_handler.MaskedHTTPStatusError: Client error '400 Bad Request' for url 'https://generativelanguage.googleapis.com/v1alpha/models/gemini-3.1-pro-preview:streamGenerateContent?alt=sse'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\streaming_handler.py", line 2133, in __anext__
    await self.fetch_stream()
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\streaming_handler.py", line 2117, in fetch_stream
    self.completion_stream = await self.make_call(client=litellm.module_level_aclient)
             ... (litellm_truncated skipped 2743 chars. Truncation is a stdout logging safeguard. Full, untruncated data is logged to logging callbacks (OTEL, Datadog, etc.) and at DEBUG level. To increase the truncation limit, set `MAX_STRING_LENGTH_STDOUT_LOG` in your env.) ...~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\streaming_handler.py", line 2355, in _log_stream_failure_and_raise
    self._handle_stream_fallback_error(e)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\streaming_handler.py", line 2452, in _handle_stream_fallback_error
    raise mapped_exception
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\streaming_handler.py", line 2415, in _handle_stream_fallback_error
    mapped_exception = exception_type(
        model=self.model,
    ...<3 lines>...
        extra_kwargs={},
    )
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\exception_mapping_utils.py", line 2675, in exception_type
    raise e  # it's already mapped
    ^^^^^^^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\exception_mapping_utils.py", line 2500, in exception_type
    _map_vertex_exception(
    ~~~~~~~~~~~~~~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<5 lines>...
        extra_information=extra_information,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\exception_mapping_utils.py", line 1252, in _map_vertex_exception
    raise BadRequestError(
    ...<11 lines>...
    )
litellm.exceptions.BadRequestError: litellm.BadRequestError: Vertex_ai_betaException BadRequestError - b'{\n  "error": {\n    "code": 400,\n    "message": "The referenced name `#/components/schemas/HTTPValidationError` in function_response.response does not match to a display_name in the function_response.parts.",\n    "status": "INVALID_ARGUMENT"\n  }\n}\n'
15:25:44 - LiteLLM Proxy:ERROR: endpoints.py:195 - litellm.proxy.proxy_server.anthropic_response(): Exception occured - litellm.BadRequestError: GeminiException BadRequestError - {
  "error": {
    "code": 400,
    "message": "The referenced name `#/components/schemas/ValidationError` in function_response.response does not match to a display_name in the function_response.parts.",
    "status": "INVALID_ARGUMENT"
  }
}
Traceback (most recent call last):
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\llms\vertex_ai\gemini\vertex_and_google_ai_studio_gemini.py", line 2835, in async_completion
    response: Final = await client.post(
                      ^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
    )
    ^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\logging_utils.py", line 300, in async_wrapper
    result: Final = await func(*args, **kwargs)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\llms\custom_httpx\http_handler.py", line 721, in post
    await _raise_masked_async_error(e, stream)
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\llms\custom_httpx\http_handler.py", line 487, in _raise_masked_async_error
    raise MaskedHTTPStatusError(e, message=_text, text=_text) from None
litellm.llms.custom_httpx.http_handler.MaskedHTTPStatusError: Client error '400 Bad Request' for url 'https://generativelanguage.googleapis.com/v1alpha/models/gemini-3.1-pro-preview:generateContent'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\main.py", line 647, in acompletion
    response = await _resolve_dispatched_chat_response(init_response)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\main.py", line 712, in _resolve_dispatched_chat_response
    return await pending
           ^^^^^^^^^^^^^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\llms\vertex_ai\gemini\vertex... (litellm_truncated skipped 2411 chars. Truncation is a stdout logging safeguard. Full, untruncated data is logged to logging callbacks (OTEL, Datadog, etc.) and at DEBUG level. To increase the truncation limit, set `MAX_STRING_LENGTH_STDOUT_LOG` in your env.) ...t litellm.acompletion(**completion_kwargs)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\utils.py", line 2065, in wrapper_async
    raise e
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\utils.py", line 1861, in wrapper_async
    result = await original_function(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\main.py", line 700, in acompletion
    raise exception_type(
          ~~~~~~~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<3 lines>...
        extra_kwargs=kwargs,
        ^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\exception_mapping_utils.py", line 2675, in exception_type
    raise e  # it's already mapped
    ^^^^^^^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\exception_mapping_utils.py", line 2500, in exception_type
    _map_vertex_exception(
    ~~~~~~~~~~~~~~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<5 lines>...
        extra_information=extra_information,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\TE000922\Ltfs_projects\Veridoc\.venv\Lib\site-packages\litellm\litellm_core_utils\exception_mapping_utils.py", line 1252, in _map_vertex_exception
    raise BadRequestError(
    ...<11 lines>...
    )
litellm.exceptions.BadRequestError: litellm.BadRequestError: GeminiException BadRequestError - {
  "error": {
    "code": 400,
    "message": "The referenced name `#/components/schemas/ValidationError` in function_response.response does not match to a display_name in the function_response.parts.",
    "status": "INVALID_ARGUMENT"
  }
}


















(veridoc) C:\Users\TE000922\Ltfs_projects\Veridoc>python C:\Users\TE000922\Ltfs_projects\Veridoc\claude\gemini_proxy.py
[INIT] Proxy listening on port 8081...
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏù[sj¸Ü%|X+ÕÀ;ÀH¥ª¸²ÃboÓäjÔ ¨3«(û£a~ítÎïñß©ó;`G¶Öc!»~Û"À+À/À,À0Ì©Ì¨À   ÀÀ" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏa´3À\¶
                k"%»ÿ
                 ÑÃ¬Ûä_ FyïN±3B'+î7pßªÉ%ôòÏØ{Öïãñý"À+À/À,À0Ì©Ì¨À        ÀÀ" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏf6öfÕÉµo¼1êaýàxºYMzÔ×%]ÛjÁZ4q¶¿)·cãcâwä?#åe¬fa="À+À/À,À0Ì©Ì¨À ÀÀ" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏ(äºØ1ÒSÈ³5Åù ÇHï÷[D[!»
                                af÷ë ±sµ\î®îÑì2ù¬SQUà<¾Æ|ÛÝ¡½7"À+À/À,À0Ì©Ì¨À    ÀÀ" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏáPöþþsÿ(\ x®ÏÒú^nØ6Ñ³+½Ú%q j§Q~ðñÄ¢/o\_>ËL
ÍÔò¢u"À+À/À,À0Ì©Ì¨À     ÀÀ" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏ·¤éðhä«êCÇz(ë«¾^[Î¦j0¹ë +|sò"ÅLê      Sî`ëZ
                                                     í"À+À/À,À0Ì©Ì¨À ÀÀ" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏ&|+Íþ<Þ½/¡®Äãö§fyG¼bO"e!@ùÅ®Ê; cº×[CÄêØ.ÐUNq8B
                                                        zÊyFì+À/À,À0Ì©Ì¨À    ÀÀ" 400 -
[HTTP] code 400, message Bad HTTP/0.9 request type ('\x16\x03\x01\x05Ó\x01\x00\x05Ï\x03\x03ÊÂPôP:\x86d^¨:éâºê\x81V¦')
[HTTP] "ÓÏÊÂPôP:d^¨:éâºêV¦Zk!Új°W" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏwÒi7¢ì¦0ðÜ½Éê¡IX@NÁeí¯çâ²K =*îÁÔDqy\ãI´Áû¡È3æ²0¯búC"À+À/À,À0Ì©Ì¨À  ÀÀ" 400 -
[HTTP] code 400, message Bad request syntax ('\x16\x03\x01\x05Ó\x01\x00\x05Ï\x03\x03Vçc\x81ySù6')
[HTTP] "ÓÏVçcySù6" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
³¤k«W¢ãÛ*C8ÏqÚdú:%I'¥3"kÌ|Û»0T{·3}èõ/!Î ás
»"À+À/À,À0Ì©Ì¨À ÀÀ" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏÎSÔÆÏ`º×ú¯þ ðº0ùì`ù¾°+ÙPÐn ¨P-£ý¼dü¾Nn(§X0µkªÙ"À+À/À,À0Ì©Ì¨À       ÀÀ" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏzßÈ÷L"o¯>KïÊñ¶µË% ãÇ6zP<Ïe
                                    4ÆoHÚ"À+À/À,À0Ì©Ì¨À ÀÀ" 400 -
[HTTP] code 400, message Bad request version ('Îo;Õì\x02\x835°\x1a\x05¿2èÅ')
[HTTP] "ÓÏQæ¡þ&
ehb^Îo;Õì5°¿2èÅ" 400 -Vv¢°NLLîvNì:¶ëåò ÔÈáò
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏ_dÀ0;Ëpj«ù·Kêß|CàsÓ¸ÜO        Êÿ¥y Åögø?K2áªH®ÅïrºLÙ?rü©"À+À/À,À0Ì©Ì¨À     ÀÀ" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏ=ÞÂl~<Lbðèº[àUqÝÂ¦|4MÈyÎ æ÷0µ¶õ+¨KcJvëà$nyÞäùò·H»*ÉÐ8e"À+À/À,À0Ì©Ì¨À       ÀÀ" 400 -
[HTTP] code 400, message Bad request syntax ('\x16\x03\x01\x05Ó\x01\x00\x05Ï\x03\x03°ôº55vnXªþæ\x8fÅuò~/\x12')
[HTTP] "ÓÏ°ôº55vnXªþæÅuò~/" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏÙë+¶JCLÌ'8ô(nuj&
                          9ñvßo$FÇß¥ Y3:        9µ©ôÐòZÁ3å52o/C#õþEuH¼"À+À/À,À0Ì©Ì¨À ÀÀ" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
[HTTP] "ÓÏ}lOÍ 8
                t¼ºòÌ!¿=
                        QhØñ¾~¨üéJæäÅÉd(|"À+À/À,À0Ì©Ì¨À ÀÀ" 400 -
[HTTP] code 400, message Bad request version ('À\x13À')
àõJK±¸ÜÊ$·Û$Þuñg57"À+À/À,À0Ì©Ì¨À£Ù¦ßWp ¹ÀÀ" 400 -


INFO:     127.0.0.1:61942 - "POST /v1/messages?beta=true HTTP/1.1" 400 Bad Request
