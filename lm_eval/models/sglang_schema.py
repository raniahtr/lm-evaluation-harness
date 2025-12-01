import copy
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.sglang_causallms import SGLangLM
from lm_eval.models.utils import (
    Collator,
    handle_stop_sequences,
    postprocess_generated_text,
)
from lm_eval.utils import simple_parse_args_string

eval_logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, ValidationError
except ImportError:  # pragma: no cover - optional dependency
    BaseModel = None
    ValidationError = None
    eval_logger.warning(
        "Pydantic not installed. Schema validation will be skipped. "
        "Install with: pip install pydantic"
    )

if TYPE_CHECKING:
    # For type checking only, BaseModel is guaranteed to be available
    # if schema_model is provided (checked at runtime)
    from pydantic import BaseModel as _PydanticBaseModel
else:
    _PydanticBaseModel = object  # Dummy type for runtime when Pydantic not installed


@register_model("sglang-schema")
class SGLangSchemaLM(SGLangLM):
    """
    Thin schema-aware wrapper around `SGLangLM`.

    This class forwards the schema through sampling params (`json_schema`) and
    optionally runs a light Pydantic validation pass.
    
    Schema Support:
    - Pydantic BaseModel classes (recommended)
    - JSON Schema files (.json):  JSON Schema specification in JSON format
    - YAML Schema files (.yaml, .yml): JSON Schema specification in YAML format
    - JSON Schema as dict or JSON/YAML string
    
   
    """
    
    def __init__(
        self,
        pretrained: str,
        schema_model: Optional[Type[_PydanticBaseModel]] = None,
        response_schema: Optional[Union[Dict, str]] = None,
        schema_file: Optional[str] = None,  
        validate_with_pydantic: bool = True,
        strict_validation: bool = False,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.schema_model = schema_model
        self.strict_validation = strict_validation
        self.validate_with_pydantic = bool(
            validate_with_pydantic and schema_model and BaseModel is not None
        )
        self.base_url = base_url or kwargs.pop("base_url", None)
        self.use_remote_api = self.base_url is not None

        if self.schema_model and BaseModel is None:
            raise ValueError(
                "schema_model requires Pydantic. Install it or set validate_with_pydantic=False."
            )

        self.response_schema = self._resolve_schema(
            schema_model=schema_model,
            response_schema=response_schema,
            schema_file=schema_file,
        )
        self._json_schema_str = (
            json.dumps(self.response_schema) if self.response_schema else None
        )

        if self._json_schema_str:
            source = (
                getattr(schema_model, "__name__", "custom")
                if schema_model
                else Path(schema_file).name
                if schema_file
                else "inline JSON"
            )
            eval_logger.info(f"Structured outputs enabled via schema '{source}'.")
        else:
            eval_logger.info("No schema provided; falling back to vanilla SGLangLM.")

        if self.use_remote_api:
            self._init_remote_backend(pretrained, kwargs)
        else:
            # Ensure device is set for local engine mode
            if "device" not in kwargs or kwargs["device"] is None:
                kwargs["device"] = "cuda"
            super().__init__(pretrained=pretrained, **kwargs)
        
    # -------------------------------------------------------------------------
    # Generation overrides
    # -------------------------------------------------------------------------
    def modify_gen_kwargs(self, kwargs: dict) -> dict:
        """Attach `json_schema` so Engine enforces `sgl.json(...)` semantics."""
        updated = super().modify_gen_kwargs(kwargs)
        if self._json_schema_str and "json_schema" not in updated:
            updated["json_schema"] = self._json_schema_str
            eval_logger.info(
                f"Added json_schema to generation kwargs (length={len(self._json_schema_str)} chars)"
            )
        elif self._json_schema_str:
            eval_logger.debug("json_schema already present in kwargs")
        return updated

    def generate_until(
        self, requests, disable_tqdm: bool = False  # type: ignore[override]
    ):
        if self.use_remote_api:
            results = self._remote_generate_until(requests, disable_tqdm=disable_tqdm)
        else:
            results = super().generate_until(requests, disable_tqdm=disable_tqdm)
        if not self.validate_with_pydantic:
            return results
        return [self._validate_output(text) for text in results]

    # -------------------------------------------------------------------------
    # Remote backend helpers
    # -------------------------------------------------------------------------
    def _init_remote_backend(self, pretrained: str, kwargs: dict):
        from lm_eval.api.model import TemplateLM

        TemplateLM.__init__(self)
        self._rank = 0
        self._max_length = kwargs.get("max_model_len") or kwargs.get("context_length")
        self._max_gen_toks = kwargs.get("max_gen_toks", 256)
        self.add_bos_token = kwargs.get("add_bos_token", False)
        self.custom_prefix_token_id = kwargs.get("prefix_token_id")
        batch_size = kwargs.get("batch_size", 1)
        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else int(batch_size)
        )
        self.think_end_token = kwargs.get("think_end_token")
        self.model = None
        self.tokenizer = self._init_remote_tokenizer(
            pretrained, kwargs.get("trust_remote_code", True)
        )
        eval_logger.info(f"Using remote SGLang server at {self.base_url}")

    def _init_remote_tokenizer(
        self, pretrained: str, trust_remote_code: bool = True
    ):
        """
        Initialize tokenizer for remote API mode.
        """
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    @property
    def eot_token_id(self):
        if self.use_remote_api:
            if hasattr(self.tokenizer, "eos_token_id"):
                return self.tokenizer.eos_token_id
            if hasattr(self.tokenizer, "tokenizer_info"):
                return self.tokenizer.tokenizer_info.get("eos_token_id")
            return None
        return super().eot_token_id

    @property
    def prefix_token_id(self):
        if self.use_remote_api:
            if self.custom_prefix_token_id is not None:
                return self.custom_prefix_token_id
            if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id:
                return self.tokenizer.bos_token_id
            return self.eot_token_id
        return super().prefix_token_id

    @property
    def max_length(self):
        if self.use_remote_api:
            if self._max_length:
                return self._max_length
            if hasattr(self.tokenizer, "model_max_length"):
                return self.tokenizer.model_max_length
            if hasattr(self.tokenizer, "tokenizer_info"):
                return self.tokenizer.tokenizer_info.get("model_max_length", 2048)
            return 2048
        return super().max_length

    @property
    def max_gen_toks(self):
        if self.use_remote_api:
            return self._max_gen_toks
        return super().max_gen_toks

    @property
    def rank(self):
        """
        Return the rank of this process.
        
        For remote API mode, we always return 0 (single process).
        For local engine mode, delegate to parent class which handles
        distributed setups properly.
        """
        if self.use_remote_api:
            return self._rank
        return super().rank

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ):
        if not self.use_remote_api:
            return super().tok_encode(
                string,
                left_truncate_len=left_truncate_len,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
            )

        if not add_special_tokens:
            add_special_tokens = self.add_bos_token

        if hasattr(self.tokenizer, "__call__"):
            encoding = self.tokenizer(
                string,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                return_attention_mask=False,
            ).input_ids
        else:
            if isinstance(string, str):
                encoding = self.tokenizer.encode(
                    string, add_special_tokens=add_special_tokens
                )
            else:
                encoding = [
                    self.tokenizer.encode(s, add_special_tokens=add_special_tokens)
                    for s in string
                ]

        if left_truncate_len:
            if isinstance(string, str):
                encoding = encoding[-left_truncate_len:]
            else:
                encoding = [enc[-left_truncate_len:] for enc in encoding]
        return encoding

    def tok_decode(self, tokens: List[int]) -> str:
        if self.use_remote_api and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(tokens)
        return super().tok_decode(tokens)

    def _remote_generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res: List[str] = []
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding = self.tok_encode(
            context, add_special_tokens=self.add_bos_token
        )
        requests_with_encoding = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            return -len(_requests[0][1]), _requests[0][0]

        re_ords = Collator(requests_with_encoding, _collate_gen, group_by=None)
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running schema-constrained generate_until requests",
        )
        eos = self.tokenizer.decode(self.eot_token_id) if self.eot_token_id else None

        for chunk in chunks:
            context_and_encoding, chunk_gen_kwargs = zip(*chunk)
            chunk_context, chunk_encoding = zip(*context_and_encoding)

            context_encoding_truncated = []
            sampling_params = []
            stop_sequences = []

            for tokens, gen_kwargs in zip(chunk_encoding, chunk_gen_kwargs):
                if not isinstance(gen_kwargs, dict):
                    raise ValueError(
                        f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                    )
                kwargs = copy.deepcopy(gen_kwargs)
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
                max_gen_toks = kwargs.pop("max_gen_toks", self.max_gen_toks)
                max_ctx_len = self.max_length - max_gen_toks

                if len(tokens) > max_ctx_len:
                    context_encoding_truncated.append(tokens[-max_ctx_len:])
                else:
                    context_encoding_truncated.append(tokens)

                kwargs = self.modify_gen_kwargs(kwargs)
                
                # When using json_schema, ensure sufficient tokens but don't override stop sequences
                # Use self._json_schema_str as source of truth (not kwargs check)
                if self._json_schema_str:
                    # Increase max tokens to ensure complete JSON generation
                    max_gen_toks = max(max_gen_toks, 512)
                    # Don't override stop sequences - let SGLang handle schema-constrained generation
                    until = []  # Clear stop sequences for schema runs
                    eval_logger.info(
                        f"Schema-constrained generation: max_new_tokens={max_gen_toks}, "
                        f"no stop sequences, json_schema_length={len(self._json_schema_str)}"
                    )
                
                # SGLang API expects max_new_tokens, not max_tokens
                # Also handle if max_tokens was already set (convert it)
                if "max_tokens" in kwargs:
                    max_gen_toks = kwargs.pop("max_tokens")
                # Remove HuggingFace-specific parameters that SGLang doesn't recognize
                kwargs.pop("skip_special_tokens", None)
                kwargs.pop("spaces_between_special_tokens", None)
                sampling_param = kwargs | {"max_new_tokens": max_gen_toks}
                # Only add stop sequences if they exist (not for schema runs)
                if until:
                    sampling_param["stop"] = until
                sampling_params.append(sampling_param)
                stop_sequences.append(until)

            cont = self._model_generate(
                requests=context_encoding_truncated,
                generate=True,
                sampling_params=sampling_params,
            )

            for output, context_str, until, gen_kwargs in zip(
                cont, chunk_context, stop_sequences, chunk_gen_kwargs
            ):
                generated_text = output.get("text", "")
                
                # When using json_schema, SGLang returns complete JSON - don't postprocess
                # (postprocessing with stop sequences can truncate JSON)
                if not self._json_schema_str:
                    generated_text = postprocess_generated_text(
                        generated_text, until, self.think_end_token
                    )
                
                res.append(generated_text)
                self.cache_hook.add_partial(
                    "generate_until", (context_str, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        sampling_params: Union[List[Dict], Dict, None] = None,
        return_logprob: bool = False,
        top_logprobs_num: int = 1,
        logprob_start_len: int = -1,
    ):
        if not self.use_remote_api:
            return super()._model_generate(
                requests=requests,
                generate=generate,
                sampling_params=sampling_params,
                return_logprob=return_logprob,
                top_logprobs_num=top_logprobs_num,
                logprob_start_len=logprob_start_len,
            )

        import requests as http_requests

        if not generate:
            sampling_params = sampling_params if sampling_params else {}
            sampling_params.update({"temperature": 0, "max_new_tokens": 1})
        if not isinstance(sampling_params, List):
            sampling_params = [sampling_params] * len(requests)

        outputs = []
        for request_tokens, sp in zip(requests, sampling_params):
            payload = {"input_ids": request_tokens, "sampling_params": sp}
            if return_logprob:
                payload.update(
                    {
                        "return_logprob": True,
                        "top_logprobs_num": top_logprobs_num,
                        "logprob_start_len": logprob_start_len,
                    }
                )
            
            # Log schema usage for debugging
            if "json_schema" in sp:
                eval_logger.info(
                    f"Sending request with json_schema (length={len(sp.get('json_schema', ''))}), "
                    f"max_new_tokens={sp.get('max_new_tokens')}, stop={sp.get('stop', 'None')}"
                )
                # Log first 200 chars of schema for debugging
                schema_preview = sp.get('json_schema', '')[:200]
                eval_logger.debug(f"json_schema preview: {schema_preview}...")
            elif self._json_schema_str:
                # This shouldn't happen - schema should be in kwargs
                eval_logger.warning(
                    f"json_schema NOT found in sampling_params but self._json_schema_str exists! "
                    f"This indicates modify_gen_kwargs didn't add it."
                )
            
            try:
                response = http_requests.post(
                    f"{self.base_url}/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                
                # Log output for debugging incomplete JSON
                if self._json_schema_str and result.get("text"):
                    text_preview = result["text"][:200]
                    text_length = len(result.get("text", ""))
                    eval_logger.info(
                        f"Received output (length={text_length}): {text_preview}..."
                    )
                    # Check if JSON is incomplete (just opening brace with whitespace)
                    text_stripped = result.get("text", "").strip()
                    if text_stripped.startswith("{") and not text_stripped.endswith("}") and text_length < 100:
                        eval_logger.warning(
                            f"Incomplete JSON output detected! Full text: {result.get('text', '')}"
                        )
                
                outputs.append(result)
            except http_requests.exceptions.HTTPError as exc:
                # Try to get error details from response
                error_detail = ""
                try:
                    error_detail = response.text
                except:
                    pass
                eval_logger.error(
                    f"Remote SGLang request failed with {response.status_code}: {exc}\n"
                    f"Error details: {error_detail}\n"
                    f"Payload keys: {list(payload.keys())}, sampling_params keys: {list(payload.get('sampling_params', {}).keys())}"
                )
                outputs.append(
                    {"text": "", "meta_info": {"input_token_logprobs": [], "input_top_logprobs": []}}
                )
            except Exception as exc:
                eval_logger.error(f"Remote SGLang request failed: {exc}")
                outputs.append(
                    {"text": "", "meta_info": {"input_token_logprobs": [], "input_top_logprobs": []}}
                )
        return outputs

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _resolve_schema(
        self,
        schema_model: Optional[Type[_PydanticBaseModel]],
        response_schema: Optional[Union[Dict, str]],
        schema_file: Optional[str],
    ) -> Optional[Dict]:
        if schema_model:
            if BaseModel is None or not issubclass(schema_model, BaseModel):
                raise ValueError("schema_model must inherit from pydantic.BaseModel")
            return schema_model.model_json_schema()

        if schema_file:
            return self._load_schema_from_path(schema_file)

        if response_schema is None:
            return None
        
        if isinstance(response_schema, dict):
            return response_schema

        if isinstance(response_schema, str):
            path_candidate = Path(response_schema)
            if path_candidate.exists():
                return self._load_schema_from_path(path_candidate)
            # Try parsing as JSON string
            try:
                return json.loads(response_schema)
            except json.JSONDecodeError:
                # Try parsing as YAML string
                try:
                    import yaml
                    parsed = yaml.safe_load(response_schema)
                    if not isinstance(parsed, dict):
                        raise ValueError(
                            "YAML string must contain a dictionary/object."
                        )
                    return parsed
                except ImportError:
                    raise ValueError(
                        f"response_schema string is not valid JSON and YAML parsing "
                        f"requires 'pyyaml'. Install with: pip install pyyaml"
                    ) from None
                except Exception as exc:
                    raise ValueError(
                        f"response_schema string is neither valid JSON, valid YAML, "
                        f"nor a file path: {exc}"
                    ) from exc

        raise ValueError(
            "response_schema must be a dict, JSON/YAML string, or file path"
        )

    @staticmethod
    def _load_schema_from_path(path_like: Union[str, Path]) -> Dict:
        """
        Load JSON Schema from a file path.
        
        Supports both JSON and YAML formats:
        - JSON files (.json): Loaded directly
        - YAML files (.yaml, .yml): Parsed and converted to dict
        
        Note: The schema must be in JSON Schema format (the specification),
        regardless of whether it's written in JSON or YAML syntax.
        
        Args:
            path_like: Path to schema file (JSON or YAML)
            
        Returns:
            Dictionary containing the JSON Schema definition
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid or not a JSON Schema object
        """
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")
        
        # Determine file format from extension
        file_ext = path.suffix.lower()
        is_yaml = file_ext in ('.yaml', '.yml')
        is_json = file_ext == '.json'
        
        # Try to load as JSON first (most common case)
        if is_json or (not is_yaml and not is_json):
            # Try JSON first (default)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("Schema file must contain a JSON object.")
                return data
            except json.JSONDecodeError:
                # If JSON parsing fails and it's not explicitly a .json file,
                # try YAML as fallback
                if is_json:
                    raise ValueError(
                        f"Schema file {path} is not valid JSON. "
                        f"Expected JSON format for .json files."
                    )
                # Fall through to YAML parsing
        
        # Try YAML parsing
        if is_yaml or (not is_json and not is_yaml):
            try:
                # Try importing yaml - make it optional
                try:
                    import yaml
                except ImportError:
                    raise ImportError(
                        "YAML file detected but 'pyyaml' is not installed. "
                        "Install it with: pip install pyyaml"
                    ) from None
                
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError(
                        "Schema file must contain a YAML object/dictionary."
                    )
                return data
            except Exception as exc:
                if is_yaml:
                    raise ValueError(
                        f"Schema file {path} is not valid YAML: {exc}"
                    ) from exc
                # If we get here, both JSON and YAML parsing failed
                raise ValueError(
                    f"Schema file {path} could not be parsed as JSON or YAML: {exc}"
                ) from exc
        
        # Should never reach here, but just in case
        raise ValueError(f"Unsupported schema file format: {path}")

    def _validate_output(self, text: str) -> str:
        """
        Validate and clean model output against Pydantic schema.
        
        SGLang with json_schema should return valid JSON directly, but we add
        fallback extraction logic to handle edge cases:
        - JSON wrapped in markdown code blocks (```json ... ```)
        - Extra text before/after JSON
        - Malformed JSON that needs cleaning
        
        Args:
            text: Raw text output from the model
            
        Returns:
            Validated JSON string (via Pydantic's model_dump_json()), or
            original text if validation fails and strict_validation=False
        """
        if not isinstance(text, str) or not text.strip():
            return text

        model_cls = self.schema_model
        if not model_cls or BaseModel is None:
            return text

        # Extract JSON from text if needed
        json_text = self._extract_json_from_text(text.strip())
        
        # Try to validate the extracted JSON
        try:
            validated = model_cls.model_validate_json(json_text)
            return validated.model_dump_json()
        except ValidationError as exc:
            # Check if JSON is incomplete (common issue with schema-constrained generation)
            if json_text.strip() in ["{", "[", "{", "["] or (
                json_text.strip().startswith("{") and not json_text.strip().endswith("}")
            ):
                eval_logger.warning(
                    f"Incomplete JSON from model (likely cut off): {repr(json_text)}. "
                    f"Raw text: {text[:100]}..."
                )
                message = f"Incomplete JSON output: {json_text}"
            else:
                message = f"Pydantic validation failed: {exc}"
        except json.JSONDecodeError as exc:
            # JSON parsing failed - this shouldn't happen with SGLang's structured output
            # but we handle it gracefully
            if json_text.strip() in ["{", "[", "{", "["]:
                message = f"Incomplete JSON output (cut off): {json_text}"
            else:
                message = f"Invalid JSON format: {exc}. Raw text: {text[:100]}..."
        except Exception as exc:  # pragma: no cover - defensive
            message = f"Unexpected validation error: {exc}"

        if self.strict_validation:
            raise ValueError(message)

        eval_logger.warning(message)
        return text

    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON string from model output, handling common formatting issues.
        
        SGLang's structured output should return clean JSON, but models sometimes
        wrap it in markdown or add extra text. This method handles:
        1. Direct JSON (most common case)
        2. JSON in markdown code blocks (```json ... ``` or ``` ... ```)
        3. JSON object/array embedded in text
        
        Args:
            text: Raw text that may contain JSON
            
        Returns:
            Extracted JSON string ready for parsing
        """
        # Clean up text: remove excessive newlines and whitespace
        # This handles cases where model generates "{ \n\n\n\n..." instead of complete JSON
        text_cleaned = text.strip()
        
        # If text is just opening brace with whitespace, it's incomplete
        if text_cleaned in ["{", "[", "{", "["] or (
            text_cleaned.startswith("{") and not text_cleaned.endswith("}") and 
            len(text_cleaned.replace("\n", "").replace(" ", "")) < 10
        ):
            eval_logger.warning(
                f"Incomplete JSON detected: {repr(text[:100])}. "
                f"This may indicate generation was cut off or model got stuck."
            )
            # Return as-is, validation will handle the error
            return text_cleaned
        
        # First, try parsing as-is (most common case with SGLang structured output)
        try:
            json.loads(text_cleaned)
            return text_cleaned
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code blocks
        # Pattern: ```json\n...\n``` or ```\n...\n```
        markdown_patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json\n...\n```
            r'```\s*\n(.*?)\n```',      # ```\n...\n``` (generic)
            r'```json\s*(.*?)```',       # ```json...``` (no newlines)
            r'```\s*(.*?)```',           # ```...``` (generic, no newlines)
        ]
        
        for pattern in markdown_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                try:
                    json.loads(extracted)
                    return extracted
                except json.JSONDecodeError:
                    continue
        
        # Try finding JSON object or array in text using balanced braces/brackets
        # This handles cases where JSON is embedded in other text
        json_obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_obj_match:
            try:
                json.loads(json_obj_match.group(0))
                return json_obj_match.group(0)
            except json.JSONDecodeError:
                pass
        
        json_arr_match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', text, re.DOTALL)
        if json_arr_match:
            try:
                json.loads(json_arr_match.group(0))
                return json_arr_match.group(0)
            except json.JSONDecodeError:
                pass
        
        # If all extraction attempts fail, return original text
        # (validation will fail, but we preserve the original for debugging)
        return text

    # -------------------------------------------------------------------------
    # CLI helper
    # -------------------------------------------------------------------------
    @classmethod
    def create_from_arg_string(
        cls, arg_string: str, additional_config: Optional[dict] = None
    ):
        args = simple_parse_args_string(arg_string)
        if additional_config:
            args.update(additional_config)
        
        for key in ("validate_with_pydantic", "strict_validation"):
            if key in args and isinstance(args[key], str):
                args[key] = args[key].lower() in {"1", "true", "yes", "on"}

        # Handle schema_model as string path (e.g., "schemas.medical_qa_schemas.StrictAnswerOnly")
        if "schema_model" in args and isinstance(args["schema_model"], str):
            schema_model_str = args["schema_model"]
            try:
                # Import the class from the string path
                module_path, class_name = schema_model_str.rsplit(".", 1)
                import importlib
                
                # Add project root to Python path if not already there
                # This allows importing modules like 'schemas.medical_qa_schemas'
                # Find the project root (directory containing 'lm_eval')
                current_file = __file__
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file))))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                module = importlib.import_module(module_path)
                args["schema_model"] = getattr(module, class_name)
                eval_logger.info(f"Imported schema model: {schema_model_str}")
            except (ImportError, AttributeError, ValueError) as e:
                raise ValueError(
                    f"Failed to import schema_model from '{schema_model_str}': {e}. "
                    f"Expected format: 'module.path.ClassName'"
                ) from e

        return cls(**args)