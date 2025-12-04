"""
言語定義の一元管理

このモジュールは、アプリケーション全体で使用される言語定義を一元管理します。
分散していた以下の定義を統合：
- config/config_loader.py: SUPPORTED_LANGUAGES, WINDOWS_LANG_MAP
- translation/translator.py: SUPPORTED_LANGUAGES, LANGUAGE_ALIASES
- gui/dialogs/settings/constants.py: TRANSCRIPTION_LANGUAGES
"""

from typing import Dict, Optional, Set, List, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LanguageInfo:
    """言語情報の完全定義"""
    code: str                    # 標準言語コード（例: "ja", "en", "zh-CN"）
    display_name: str            # ローカライズ表示名（例: "日本語"）
    english_name: str            # 英語表示名（例: "Japanese"）
    native_name: str            # ネイティブ表示名（例: "日本語"）
    flag: str                   # 国旗絵文字（例: "🇯🇵"）
    iso639_1: Optional[str]     # ISO 639-1コード（例: "ja"）
    iso639_3: Optional[str]     # ISO 639-3コード（例: "jpn"）
    windows_lcid: Optional[int] # Windows言語ID（例: 0x0411）
    google_code: Optional[str]  # Google翻訳コード（例: "ja"）
    translation_code: str       # 翻訳API用コード（例: "ja"）
    asr_code: str              # ASR用コード（例: "ja"）
    supported_engines: List[str] # 対応エンジンリスト
    translation_services: List[str] = field(default_factory=list)  # 対応翻訳サービスリスト


class Languages:
    """
    アプリケーション全体の言語定義マスタークラス

    統合元：
    - config/config_loader.py: SUPPORTED_LANGUAGES, WINDOWS_LANG_MAP
    - translation/translator.py: SUPPORTED_LANGUAGES, LANGUAGE_ALIASES
    - gui/dialogs/settings/constants.py: TRANSCRIPTION_LANGUAGES
    """

    # 特殊コード定義
    AUTO = "auto"  # 自動検出用の定数
    _UNKNOWN_WARNING_LIMIT = 5
    _unknown_codes_logged: Set[str] = set()
    _unknown_log_suppressed = False

    # ========== マスター定義（すべての情報を統合） ==========
    _LANGUAGES: Dict[str, LanguageInfo] = {
        "ja": LanguageInfo(
            code="ja",
            display_name="日本語",
            english_name="Japanese",
            native_name="日本語",
            flag="🇯🇵",
            iso639_1="ja",
            iso639_3="jpn",
            windows_lcid=0x0411,
            google_code="ja",
            translation_code="ja",
            asr_code="ja",
            supported_engines=["reazonspeech", "whispers2t", "parakeet_ja"],
            translation_services=["google", "riva"]
        ),
        "en": LanguageInfo(
            code="en",
            display_name="English",
            english_name="English",
            native_name="English",
            flag="🇺🇸",
            iso639_1="en",
            iso639_3="eng",
            windows_lcid=0x0409,
            google_code="en",
            translation_code="en",
            asr_code="en",
            supported_engines=["parakeet", "whispers2t", "canary", "voxtral"],
            translation_services=["google", "riva"]
        ),
        "zh-CN": LanguageInfo(
            code="zh-CN",
            display_name="中文(简体)",
            english_name="Simplified Chinese",
            native_name="简体中文",
            flag="🇨🇳",
            iso639_1="zh",
            iso639_3="zho",
            windows_lcid=0x0804,
            google_code="zh-CN",
            translation_code="zh-CN",
            asr_code="zh",
            supported_engines=["whispers2t"],
            translation_services=["google", "riva"]
        ),
        "zh-TW": LanguageInfo(
            code="zh-TW",
            display_name="中文(繁體)",
            english_name="Traditional Chinese",
            native_name="繁體中文",
            flag="🇹🇼",
            iso639_1="zh",
            iso639_3="zho",
            windows_lcid=0x0404,
            google_code="zh-TW",
            translation_code="zh-TW",
            asr_code="zh",
            supported_engines=["whispers2t"],
            translation_services=["google", "riva"]
        ),
        "ko": LanguageInfo(
            code="ko",
            display_name="한국어",
            english_name="Korean",
            native_name="한국어",
            flag="🇰🇷",
            iso639_1="ko",
            iso639_3="kor",
            windows_lcid=0x0412,
            google_code="ko",
            translation_code="ko",
            asr_code="ko",
            supported_engines=["whispers2t"],
            translation_services=["google", "riva"]
        ),
        "de": LanguageInfo(
            code="de",
            display_name="Deutsch",
            english_name="German",
            native_name="Deutsch",
            flag="🇩🇪",
            iso639_1="de",
            iso639_3="deu",
            windows_lcid=0x0407,
            google_code="de",
            translation_code="de",
            asr_code="de",
            supported_engines=["whispers2t", "canary", "voxtral"],
            translation_services=["google", "riva"]
        ),
        "fr": LanguageInfo(
            code="fr",
            display_name="Français",
            english_name="French",
            native_name="Français",
            flag="🇫🇷",
            iso639_1="fr",
            iso639_3="fra",
            windows_lcid=0x040C,
            google_code="fr",
            translation_code="fr",
            asr_code="fr",
            supported_engines=["whispers2t", "canary", "voxtral"],
            translation_services=["google", "riva"]
        ),
        "es": LanguageInfo(
            code="es",
            display_name="Español",
            english_name="Spanish",
            native_name="Español",
            flag="🇪🇸",
            iso639_1="es",
            iso639_3="spa",
            windows_lcid=0x0C0A,
            google_code="es",
            translation_code="es",
            asr_code="es",
            supported_engines=["whispers2t", "canary", "voxtral"],
            translation_services=["google", "riva"]
        ),
        "es-ES": LanguageInfo(
            code="es-ES",
            display_name="Español (España)",
            english_name="European Spanish",
            native_name="Español (España)",
            flag="🇪🇸",
            iso639_1="es",
            iso639_3="spa",
            windows_lcid=0x0C0A,
            google_code="es",
            translation_code="es-ES",
            asr_code="es",
            supported_engines=["riva"],
            translation_services=["riva"]
        ),
        "es-US": LanguageInfo(
            code="es-US",
            display_name="Español (Latinoamérica)",
            english_name="Latin American Spanish",
            native_name="Español (Latinoamérica)",
            flag="🇲🇽",
            iso639_1="es",
            iso639_3="spa",
            windows_lcid=0x540A,
            google_code="es",
            translation_code="es-US",
            asr_code="es",
            supported_engines=["riva"],
            translation_services=["riva"]
        ),
        "ru": LanguageInfo(
            code="ru",
            display_name="Русский",
            english_name="Russian",
            native_name="Русский",
            flag="🇷🇺",
            iso639_1="ru",
            iso639_3="rus",
            windows_lcid=0x0419,
            google_code="ru",
            translation_code="ru",
            asr_code="ru",
            supported_engines=["whispers2t"],
            translation_services=["google", "riva"]
        ),
        "ar": LanguageInfo(
            code="ar",
            display_name="العربية",
            english_name="Arabic",
            native_name="العربية",
            flag="🇸🇦",
            iso639_1="ar",
            iso639_3="ara",
            windows_lcid=0x0401,
            google_code="ar",
            translation_code="ar",
            asr_code="ar",
            supported_engines=["whispers2t"],
            translation_services=["google", "riva"]
        ),
        "pt": LanguageInfo(
            code="pt",
            display_name="Português",
            english_name="Portuguese",
            native_name="Português",
            flag="🇵🇹",
            iso639_1="pt",
            iso639_3="por",
            windows_lcid=0x0816,
            google_code="pt",
            translation_code="pt",
            asr_code="pt",
            supported_engines=["whispers2t", "voxtral"],
            translation_services=["google", "riva"]
        ),
        "pt-BR": LanguageInfo(
            code="pt-BR",
            display_name="Português (Brasil)",
            english_name="Brazilian Portuguese",
            native_name="Português (Brasil)",
            flag="🇧🇷",
            iso639_1="pt",
            iso639_3="por",
            windows_lcid=0x0416,
            google_code="pt",
            translation_code="pt-BR",
            asr_code="pt",
            supported_engines=["riva"],
            translation_services=["riva"]
        ),
        "it": LanguageInfo(
            code="it",
            display_name="Italiano",
            english_name="Italian",
            native_name="Italiano",
            flag="🇮🇹",
            iso639_1="it",
            iso639_3="ita",
            windows_lcid=0x0410,
            google_code="it",
            translation_code="it",
            asr_code="it",
            supported_engines=["whispers2t", "voxtral"],
            translation_services=["google"]
        ),
        "hi": LanguageInfo(
            code="hi",
            display_name="हिन्दी",
            english_name="Hindi",
            native_name="हिन्दी",
            flag="🇮🇳",
            iso639_1="hi",
            iso639_3="hin",
            windows_lcid=0x0439,
            google_code="hi",
            translation_code="hi",
            asr_code="hi",
            supported_engines=["whispers2t", "voxtral"],
            translation_services=["google"]
        ),
        "nl": LanguageInfo(
            code="nl",
            display_name="Nederlands",
            english_name="Dutch",
            native_name="Nederlands",
            flag="🇳🇱",
            iso639_1="nl",
            iso639_3="nld",
            windows_lcid=0x0413,
            google_code="nl",
            translation_code="nl",
            asr_code="nl",
            supported_engines=["whispers2t", "voxtral"],
            translation_services=["google"]
        ),
    }

    # ========== エイリアス定義（正規化用） ==========
    _ALIASES: Dict[str, str] = {
        # 短縮形 → 標準形（小文字で定義）
        "zh": "zh-CN",       # 中国語デフォルトは簡体字
        "cn": "zh-CN",       # 簡体字
        "tw": "zh-TW",       # 繁体字
        "hk": "zh-TW",       # 香港は繁体字
        "zh-hk": "zh-TW",    # zh-HKも繁体字へ
        "zh-hans": "zh-CN",  # 簡体字（別名）
        "zh-hant": "zh-TW",  # 繁体字（別名）
        "en-us": "en",       # 米国英語
        "en-gb": "en",       # 英国英語
    }

    # ========== Google翻訳レガシーコード ==========
    _GOOGLE_LEGACY_CODES: Dict[str, str] = {
        "he": "iw",  # Hebrew: ISO標準 → Googleレガシー
        "jv": "jw",  # Javanese: ISO標準 → Googleレガシー
    }

    # ========== Windows LCID逆引きマップ ==========
    _LCID_TO_CODE: Dict[int, str] = {
        info.windows_lcid: code
        for code, info in _LANGUAGES.items()
        if info.windows_lcid
    }

    # ==================== パブリックAPI ====================

    @classmethod
    def normalize(cls, code: str) -> Optional[str]:
        """
        言語コードを正規化する
        大文字小文字を適切に処理しつつ、地域コードを保持

        Args:
            code: 入力言語コード（"ja", "JA", "zh-TW", "auto"等）

        Returns:
            正規化された言語コード、または None

        Examples:
            >>> Languages.normalize("JA")
            "ja"
            >>> Languages.normalize("zh-TW")
            "zh-TW"  # 繁体字は保持
            >>> Languages.normalize("zh")
            "zh-CN"  # デフォルトは簡体字
            >>> Languages.normalize("auto")
            "auto"   # 特殊コードはそのまま
        """
        if not code:
            return None

        # 特殊コード "auto" はそのまま返す
        code_lower = code.lower().strip()
        if code_lower == cls.AUTO:
            return cls.AUTO

        # 完全一致を優先（大文字小文字無視）
        for standard_code in cls._LANGUAGES.keys():
            if standard_code.lower() == code_lower:
                return standard_code  # 正式な形（zh-CN, zh-TW等）を返す

        # エイリアスチェック（小文字で比較）
        for alias_key, target in cls._ALIASES.items():
            if alias_key == code_lower:
                return target

        # セパレータ付きコードの特別処理
        # ただし、地域指定がある場合は安易に分割しない
        base_code = code_lower  # fallback for codes without separators
        for sep in ["-", "_"]:
            if sep in code_lower:
                # まず全体でエイリアスを再チェック（アンダースコアをハイフンに正規化）
                normalized_full = code_lower.replace("_", "-")
                for alias_key, target in cls._ALIASES.items():
                    if alias_key == normalized_full:
                        return target

                # ベースコードと地域コードに分割
                parts = code_lower.split(sep)
                if len(parts) >= 2:
                    base_code = parts[0]
                    region_code = parts[1]

                    # zh_tw → tw, zh_cn → cn のエイリアスチェック
                    # 地域コードでエイリアスを検索
                    for alias_key, target in cls._ALIASES.items():
                        if alias_key == region_code:
                            # twやcnがエイリアスとして見つかれば、そのターゲットを返す
                            return target

                # ベースコードで検索
                base_code = parts[0]
                for standard_code in cls._LANGUAGES.keys():
                    if standard_code.lower() == base_code:
                        return standard_code

        # ベースコードでエイリアスを検索
        for alias_key, target in cls._ALIASES.items():
            if alias_key == base_code:
                return target

        # 未知コードのログレベルを抑制（初回のみ警告、それ以降はデバッグ）
        normalized_code = code_lower
        if normalized_code not in cls._unknown_codes_logged:
            cls._unknown_codes_logged.add(normalized_code)
            if len(cls._unknown_codes_logged) <= cls._UNKNOWN_WARNING_LIMIT:
                logger.warning(f"Unknown language code: {code}")
            elif not cls._unknown_log_suppressed:
                logger.warning(
                    "Unknown language code detected (example: %s). "
                    "Further messages will be suppressed.",
                    code
                )
                cls._unknown_log_suppressed = True
            else:
                logger.debug(f"Unknown language code suppressed: {code}")
        else:
            logger.debug(f"Unknown language code repeated: {code}")

        return None

    @classmethod
    def is_auto(cls, code: str) -> bool:
        """自動検出モードかチェック"""
        return code and code.lower() == cls.AUTO

    @classmethod
    def get_info(cls, code: str) -> Optional[LanguageInfo]:
        """
        完全な言語情報を取得

        Args:
            code: 言語コード

        Returns:
            LanguageInfo オブジェクト、またはNone（autoや不明なコードの場合）
        """
        if cls.is_auto(code):
            return None  # autoの場合はNone

        normalized = cls.normalize(code)
        return cls._LANGUAGES.get(normalized) if normalized else None

    @classmethod
    def get_display_name(cls, code: str, english: bool = False) -> str:
        """
        表示名を取得

        Args:
            code: 言語コード
            english: Trueの場合は英語名を返す

        Returns:
            表示名（見つからない場合は元のコードを返す）
        """
        info = cls.get_info(code)
        if info:
            return info.english_name if english else info.display_name
        return code

    @classmethod
    def get_google_code(cls, code: str) -> str:
        """
        Google翻訳用コードを取得（レガシーコード適用済み）

        Args:
            code: 言語コード

        Returns:
            Google翻訳用コード
        """
        if cls.is_auto(code):
            return cls.AUTO

        info = cls.get_info(code)
        if info and info.google_code:
            # レガシーコード変換
            return cls._GOOGLE_LEGACY_CODES.get(info.google_code, info.google_code)
        return code

    @classmethod
    def from_windows_lcid(cls, lcid: int) -> Optional[str]:
        """
        Windows言語IDから言語コードを取得

        Args:
            lcid: Windows言語ID

        Returns:
            言語コード、またはNone
        """
        return cls._LCID_TO_CODE.get(lcid)

    @classmethod
    def get_supported_codes(cls) -> Set[str]:
        """サポート言語コードのセットを取得"""
        return set(cls._LANGUAGES.keys())

    @classmethod
    def get_engines_for_language(cls, code: str) -> List[str]:
        """
        指定言語をサポートするエンジンリストを取得

        Args:
            code: 言語コード

        Returns:
            エンジンIDのリスト
        """
        info = cls.get_info(code)
        return info.supported_engines if info else []

    # ==================== 非推奨API（削除予定） ====================

    @classmethod
    def get_translation_languages_dict(cls) -> Dict[str, str]:
        """
        翻訳用言語辞書を取得（非推奨）

        .. deprecated:: 2.1.0
           代わりに get_languages_for_translation_service() を使用してください
        """
        import warnings
        warnings.warn(
            "get_translation_languages_dict() is deprecated. "
            "Use get_languages_for_translation_service() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return {code: info.display_name for code, info in cls._LANGUAGES.items()}

    @classmethod
    def get_transcription_languages_dict(cls) -> Dict[str, Dict[str, str]]:
        """
        文字起こし用言語辞書を取得（非推奨）

        .. deprecated:: 2.1.0
           代わりに get_info() または get_display_name() を使用してください
        """
        import warnings
        warnings.warn(
            "get_transcription_languages_dict() is deprecated. "
            "Use get_info() or get_display_name() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return {
            code: {
                "name": info.display_name,
                "native": info.native_name,
                "flag": info.flag,
            }
            for code, info in cls._LANGUAGES.items()
        }

    @classmethod
    def get_all_codes(cls) -> List[str]:
        """
        すべての言語コードを取得

        Returns:
            言語コードのリスト（autoを含む）
        """
        codes = list(cls._LANGUAGES.keys())
        codes.append(cls.AUTO)
        return codes

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """
        言語エイリアス辞書を取得

        Returns:
            エイリアス -> 正規化されたコードのマッピング
        """
        return cls._ALIASES.copy()

    @classmethod
    def get_name(cls, code: str) -> Optional[str]:
        """
        言語の表示名を取得

        Args:
            code: 言語コード

        Returns:
            表示名、見つからない場合はNone
        """
        info = cls.get_info(code)
        return info.display_name if info else None

    @classmethod
    def is_valid(cls, code: str) -> bool:
        """
        言語コードが有効かどうかを判定

        Args:
            code: 言語コード

        Returns:
            有効な場合True
        """
        if not code:
            return False
        normalized = cls.normalize(code)
        return normalized is not None

    @classmethod
    def get_languages_for_translation_service(cls, service: str) -> List[Tuple[str, str]]:
        """
        指定された翻訳サービスがサポートする言語リストを取得

        Args:
            service: 翻訳サービス名（'google', 'riva'等）

        Returns:
            [(code, display_name), ...] の形式のリスト
            注: ローカライゼーションはUI側で行う

        Note:
            - translation_servicesメタデータからサポート言語を取得
            - バックエンドのインポート不要（依存関係の正常化）
            - 純粋なメタデータ操作のみ
        """
        result = []

        # メタデータから直接サポート言語を取得（バックエンドインポート不要）
        for code, info in cls._LANGUAGES.items():
            if service in info.translation_services:
                result.append((code, info.display_name))

        return result
