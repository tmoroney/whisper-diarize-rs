pub fn round_to_places(value: f64, places: i32) -> f64 {
    let factor = 10f64.powi(places);
    (value * factor).round() / factor
}

// Convert centiseconds to seconds (1 centisecond = 10ms)
pub fn cs_to_s(cs: i64) -> f64 {
    cs as f64 * 0.01
}

/// List of supported target language codes for Google Translate (unofficial endpoint).
pub fn get_translate_languages() -> Vec<&'static str> {
    vec![
        "af", "sq", "am", "ar", "hy", "az", "eu", "be", "bn", "bs", "bg", "ca", "ceb", "ny", "zh", "zh-TW",
        "co", "hr", "cs", "da", "nl", "en", "eo", "et", "tl", "fi", "fr", "fy", "gl", "ka", "de", "el", "gu",
        "ht", "ha", "haw", "he", "hi", "hmn", "hu", "is", "ig", "id", "ga", "it", "ja", "jv", "kn", "kk", "km",
        "rw", "ko", "ku", "ky", "lo", "la", "lv", "lt", "lb", "mk", "mg", "ms", "ml", "mt", "mi", "mr", "mn",
        "my", "ne", "no", "or", "ps", "fa", "pl", "pt", "pa", "ro", "ru", "sm", "gd", "sr", "st", "sn", "sd",
        "si", "sk", "sl", "so", "es", "su", "sw", "sv", "tg", "ta", "te", "th", "tr", "uk", "ur", "ug", "uz",
        "vi", "cy", "xh", "yi", "yo", "zu",
    ]
}

/// List of Whisper-supported language codes (including "auto").
pub fn get_whisper_languages() -> Vec<&'static str> {
    vec![
        // Auto detection
        "auto",
        // Core set
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id",
        "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg",
        "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br",
        "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so",
        "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt",
        "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
    ]
}

// List of supported language codes for Whisper (includes "auto"):
// - `auto`: Automatic language detection
// - `en`: English
// - `zh`: Chinese
// - `de`: German
// - `es`: Spanish
// - `ru`: Russian
// - `ko`: Korean
// - `fr`: French
// - `ja`: Japanese
// - `pt`: Portuguese
// - `tr`: Turkish
// - `pl`: Polish
// - `ca`: Catalan
// - `nl`: Dutch
// - `ar`: Arabic
// - `sv`: Swedish
// - `it`: Italian
// - `id`: Indonesian
// - `hi`: Hindi
// - `fi`: Finnish
// - `vi`: Vietnamese
// - `he`: Hebrew
// - `uk`: Ukrainian
// - `el`: Greek
// - `ms`: Malay
// - `cs`: Czech
// - `ro`: Romanian
// - `da`: Danish
// - `hu`: Hungarian
// - `ta`: Tamil
// - `no`: Norwegian
// - `th`: Thai
// - `ur`: Urdu
// - `hr`: Croatian
// - `bg`: Bulgarian
// - `lt`: Lithuanian
// - `la`: Latin
// - `lv`: Latvian
// - `mt`: Maltese
// - `mi`: Maori
// - `mr`: Marathi
// - `mn`: Mongolian
// - `my`: Myanmar (Burmese)
// - `ne`: Nepali
// - `so`: Somali
// - `es`: Spanish
// - `su`: Sundanese
// - `sw`: Swahili
// - `sv`: Swedish
// - `tg`: Tajik
// - `ta`: Tamil
// - `te`: Telugu
// - `th`: Thai
// - `tr`: Turkish
// - `uk`: Ukrainian
// - `ur`: Urdu
// - `ug`: Uyghur
// - `uz`: Uzbek
// - `vi`: Vietnamese
// - `cy`: Welsh
// - `xh`: Xhosa
// - `yi`: Yiddish
// - `yo`: Yoruba
// - `zu`: Zulu

// List of supported language codes for translation:
// - `af`: Afrikaans
// - `sq`: Albanian
// - `am`: Amharic
// - `ar`: Arabic
// - `hy`: Armenian
// - `az`: Azerbaijani
// - `eu`: Basque
// - `be`: Belarusian
// - `bn`: Bengali
// - `bs`: Bosnian
// - `bg`: Bulgarian
// - `ca`: Catalan
// - `ceb`: Cebuano
// - `ny`: Chichewa
// - `zh`: Chinese (Simplified)
// - `zh-TW`: Chinese (Traditional)
// - `co`: Corsican
// - `hr`: Croatian
// - `cs`: Czech
// - `da`: Danish
// - `nl`: Dutch
// - `en`: English
// - `eo`: Esperanto
// - `et`: Estonian
// - `tl`: Filipino
// - `fi`: Finnish
// - `fr`: French
// - `fy`: Frisian
// - `gl`: Galician
// - `ka`: Georgian
// - `de`: German
// - `el`: Greek
// - `gu`: Gujarati
// - `ht`: Haitian Creole
// - `ha`: Hausa
// - `haw`: Hawaiian
// - `he`: Hebrew
// - `hi`: Hindi
// - `hmn`: Hmong
// - `hu`: Hungarian
// - `is`: Icelandic
// - `ig`: Igbo
// - `id`: Indonesian
// - `ga`: Irish
// - `it`: Italian
// - `ja`: Japanese
// - `jv`: Javanese
// - `kn`: Kannada
// - `kk`: Kazakh
// - `km`: Khmer
// - `rw`: Kinyarwanda
// - `ko`: Korean
// - `ku`: Kurdish (Kurmanji)
// - `ky`: Kyrgyz
// - `lo`: Lao
// - `la`: Latin
// - `lv`: Latvian
// - `lt`: Lithuanian
// - `lb`: Luxembourgish
// - `mk`: Macedonian
// - `mg`: Malagasy
// - `ms`: Malay
// - `ml`: Malayalam
// - `mt`: Maltese
// - `mi`: Maori
// - `mr`: Marathi
// - `mn`: Mongolian
// - `my`: Myanmar (Burmese)
// - `ne`: Nepali
// - `no`: Norwegian
// - `or`: Odia (Oriya)
// - `ps`: Pashto
// - `fa`: Persian
// - `pl`: Polish
// - `pt`: Portuguese
// - `pa`: Punjabi
// - `ro`: Romanian
// - `ru`: Russian
// - `sm`: Samoan
// - `gd`: Scots Gaelic
// - `sr`: Serbian
// - `st`: Sesotho
// - `sn`: Shona
// - `sd`: Sindhi
// - `si`: Sinhala
// - `sk`: Slovak
// - `sl`: Slovenian
// - `so`: Somali
// - `es`: Spanish
// - `su`: Sundanese
// - `sw`: Swahili
// - `sv`: Swedish
// - `tg`: Tajik
// - `ta`: Tamil
// - `te`: Telugu
// - `th`: Thai
// - `tr`: Turkish
// - `uk`: Ukrainian
// - `ur`: Urdu
// - `ug`: Uyghur
// - `uz`: Uzbek
// - `vi`: Vietnamese
// - `cy`: Welsh
// - `xh`: Xhosa
// - `yi`: Yiddish
// - `yo`: Yoruba
// - `zu`: Zulu