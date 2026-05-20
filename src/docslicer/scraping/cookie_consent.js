() => {
    function isBlocking() {
        const bodyStyle = document.body.style;
        const documentStyle = document.documentElement.style;
        if (bodyStyle.overflow === 'hidden' || documentStyle.overflow === 'hidden') return true;

        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        for (const el of document.querySelectorAll('*')) {
            const computedStyle = getComputedStyle(el);
            if (computedStyle.position !== 'fixed' && computedStyle.position !== 'sticky') continue;
            if (
                computedStyle.display === 'none' ||
                computedStyle.visibility === 'hidden' ||
                computedStyle.opacity === '0'
            ) continue;
            if ((parseInt(computedStyle.zIndex) || 0) < 10) continue;

            const rect = el.getBoundingClientRect();
            if (rect.width * rect.height > viewportWidth * viewportHeight * 0.4) return true;
        }
        return false;
    }

    if (!isBlocking()) return false;

    const selectors = [
        '#onetrust-accept-btn-handler',
        '#accept-recommended-btn-handler',
        '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
        '#CybotCookiebotDialogBodyButtonAccept',
        '.qc-cmp2-summary-buttons button:first-child',
        '[data-testid="accept-all"]',
        '[data-testid="cookie-accept"]',
        '[aria-label*="Accept"]',
        '[aria-label*="accept"]',
    ];
    for (const selector of selectors) {
        const el = document.querySelector(selector);
        if (el && el.offsetParent !== null) {
            el.click();
            return true;
        }
    }

    const acceptTexts = [
        'accept all',
        'accept cookies',
        'allow all',
        'allow cookies',
        'agree',
        'i agree',
        'got it',
        'ok, i agree',
    ];
    for (const el of document.querySelectorAll('button, [role="button"]')) {
        const text = el.textContent.trim().toLowerCase();
        if (acceptTexts.includes(text) && el.offsetParent !== null) {
            el.click();
            return true;
        }
    }
    return false;
}
