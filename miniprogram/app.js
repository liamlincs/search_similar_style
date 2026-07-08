function isBareTimeoutError(value) {
  const message = String(
    (value && value.message) ||
    (value && value.reason && value.reason.message) ||
    value ||
    ""
  );
  const stack = String(
    (value && value.stack) ||
    (value && value.reason && value.reason.stack) ||
    ""
  );
  return /^timeout$/i.test(message.trim()) && (!stack || /WAServiceMainContext|appservice\/__dev__/.test(stack));
}

App({
  onError(err) {
    if (isBareTimeoutError(err)) {
      console.warn("[app:onError:ignored-timeout]", err);
      return;
    }
    console.error("[app:onError]", err);
  },
  onUnhandledRejection(res) {
    if (isBareTimeoutError(res)) {
      console.warn("[app:onUnhandledRejection:ignored-timeout]", res);
      return;
    }
    console.error("[app:onUnhandledRejection]", res);
  },
  onPageNotFound(res) {
    console.error("[app:onPageNotFound]", res);
  }
})
